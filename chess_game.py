#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRANDMASTER CHESS  -  complete single-file chess game with an AI opponent
=========================================================================

Run the game :      python chess_game.py
Run as UCI engine : python chess_game.py --uci      (plug into Arena / CuteChess / Lichess-bot)
Move-gen test :     python chess_game.py --perft 4

Requires:  pip install pygame        (only for the GUI; the engine itself is pure Python)

Features
--------
* Full rules: castling, en passant, promotion, check / checkmate / stalemate,
  threefold repetition, fifty-move rule, insufficient material
* Engine: 0x88 board, incremental Zobrist hashing + incremental evaluation,
  iterative deepening alpha-beta (PVS), transposition table, null-move pruning,
  late-move reductions, killer + history heuristics, MVV-LVA ordering,
  quiescence search, check extensions, tapered piece-square evaluation
* 4 difficulty levels, hints, undo, board flip, 4 board themes
* Procedurally drawn anti-aliased pieces (no image files needed), smooth animations,
  drag & drop or click-to-move, legal-move dots, last-move + check highlights,
  evaluation bar, captured pieces, SAN move list, PGN export, synthesized sounds

Keys:  N new game | U / Ctrl+Z undo | F flip | H hint | T theme | M sound | S save PGN | Esc cancel
"""
import sys
import os
import time
import math
import random
import threading
import array
import datetime

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
try:
    import pygame
except ImportError:  # the engine / UCI mode works without pygame
    pygame = None

# =============================================================================
#                                   ENGINE
# =============================================================================
WHITE, BLACK = 0, 1
PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 1, 2, 3, 4, 5, 6
START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
SQ64 = [r * 16 + f for r in range(8) for f in range(8)]
KN_D = (-33, -31, -18, -14, 14, 18, 31, 33)
B_D = (-17, -15, 15, 17)
R_D = (-16, -1, 1, 16)
K_D = B_D + R_D
F_DOUBLE, F_EP, F_CASTLE = 1, 2, 3
INF = 10 ** 6
MATE = 30000

CMASK = [15] * 128
CMASK[0] &= ~2
CMASK[7] &= ~1
CMASK[4] &= ~3
CMASK[112] &= ~8
CMASK[119] &= ~4
CMASK[116] &= ~12

_rng = random.Random(0xC0FFEE)
ZP = [[_rng.getrandbits(64) for _ in range(128)] for _ in range(16)]
ZSIDE = _rng.getrandbits(64)
ZCASTLE = [_rng.getrandbits(64) for _ in range(16)]
ZEP = [_rng.getrandbits(64) for _ in range(8)]

MAT = [0, 100, 320, 330, 500, 900, 0]
PHASE = [0, 0, 1, 1, 2, 4, 0]

_T_PAWN = [0, 0, 0, 0, 0, 0, 0, 0,
           50, 50, 50, 50, 50, 50, 50, 50,
           10, 10, 20, 30, 30, 20, 10, 10,
           5, 5, 10, 25, 25, 10, 5, 5,
           0, 0, 0, 20, 20, 0, 0, 0,
           5, -5, -10, 0, 0, -10, -5, 5,
           5, 10, 10, -20, -20, 10, 10, 5,
           0, 0, 0, 0, 0, 0, 0, 0]
_T_KNIGHT = [-50, -40, -30, -30, -30, -30, -40, -50,
             -40, -20, 0, 0, 0, 0, -20, -40,
             -30, 0, 10, 15, 15, 10, 0, -30,
             -30, 5, 15, 20, 20, 15, 5, -30,
             -30, 0, 15, 20, 20, 15, 0, -30,
             -30, 5, 10, 15, 15, 10, 5, -30,
             -40, -20, 0, 5, 5, 0, -20, -40,
             -50, -40, -30, -30, -30, -30, -40, -50]
_T_BISHOP = [-20, -10, -10, -10, -10, -10, -10, -20,
             -10, 0, 0, 0, 0, 0, 0, -10,
             -10, 0, 5, 10, 10, 5, 0, -10,
             -10, 5, 5, 10, 10, 5, 5, -10,
             -10, 0, 10, 10, 10, 10, 0, -10,
             -10, 10, 10, 10, 10, 10, 10, -10,
             -10, 5, 0, 0, 0, 0, 5, -10,
             -20, -10, -10, -10, -10, -10, -10, -20]
_T_ROOK = [0, 0, 0, 0, 0, 0, 0, 0,
           5, 10, 10, 10, 10, 10, 10, 5,
           -5, 0, 0, 0, 0, 0, 0, -5,
           -5, 0, 0, 0, 0, 0, 0, -5,
           -5, 0, 0, 0, 0, 0, 0, -5,
           -5, 0, 0, 0, 0, 0, 0, -5,
           -5, 0, 0, 0, 0, 0, 0, -5,
           0, 0, 0, 5, 5, 0, 0, 0]
_T_QUEEN = [-20, -10, -10, -5, -5, -10, -10, -20,
            -10, 0, 0, 0, 0, 0, 0, -10,
            -10, 0, 5, 5, 5, 5, 0, -10,
            -5, 0, 5, 5, 5, 5, 0, -5,
            0, 0, 5, 5, 5, 5, 0, -5,
            -10, 5, 5, 5, 5, 5, 0, -10,
            -10, 0, 5, 0, 0, 0, 0, -10,
            -20, -10, -10, -5, -5, -10, -10, -20]
_T_KING_MG = [-30, -40, -40, -50, -50, -40, -40, -30,
              -30, -40, -40, -50, -50, -40, -40, -30,
              -30, -40, -40, -50, -50, -40, -40, -30,
              -30, -40, -40, -50, -50, -40, -40, -30,
              -20, -30, -30, -40, -40, -30, -30, -20,
              -10, -20, -20, -20, -20, -20, -20, -10,
              20, 20, 0, 0, 0, 0, 20, 20,
              20, 30, 10, 0, 0, 10, 30, 20]
_T_KING_EG = [-50, -40, -30, -20, -20, -30, -40, -50,
              -30, -20, -10, 0, 0, -10, -20, -30,
              -30, -10, 20, 30, 30, 20, -10, -30,
              -30, -10, 30, 40, 40, 30, -10, -30,
              -30, -10, 30, 40, 40, 30, -10, -30,
              -30, -10, 20, 30, 30, 20, -10, -30,
              -30, -30, 0, 0, 0, 0, -30, -30,
              -50, -30, -30, -30, -30, -30, -30, -50]
_T_PAWN_EG = [0, 0, 0, 0, 0, 0, 0, 0,
              80, 80, 80, 80, 80, 80, 80, 80,
              50, 50, 50, 50, 50, 50, 50, 50,
              30, 30, 30, 30, 30, 30, 30, 30,
              15, 15, 15, 20, 20, 15, 15, 15,
              5, 5, 5, 10, 10, 5, 5, 5,
              0, 0, 0, -10, -10, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0]

_TAB_MG = {1: _T_PAWN, 2: _T_KNIGHT, 3: _T_BISHOP, 4: _T_ROOK, 5: _T_QUEEN, 6: _T_KING_MG}
_TAB_EG = {1: _T_PAWN_EG, 2: _T_KNIGHT, 3: _T_BISHOP, 4: _T_ROOK, 5: _T_QUEEN, 6: _T_KING_EG}
PST_MG = [[0] * 128 for _ in range(16)]
PST_EG = [[0] * 128 for _ in range(16)]
for _t in range(1, 7):
    for _r in range(8):
        for _f in range(8):
            _sq = _r * 16 + _f
            _wi, _bi = (7 - _r) * 8 + _f, _r * 8 + _f
            PST_MG[_t][_sq] = MAT[_t] + _TAB_MG[_t][_wi]
            PST_EG[_t][_sq] = MAT[_t] + _TAB_EG[_t][_wi]
            PST_MG[_t | 8][_sq] = -(MAT[_t] + _TAB_MG[_t][_bi])
            PST_EG[_t | 8][_sq] = -(MAT[_t] + _TAB_EG[_t][_bi])

FILES = "abcdefgh"
PIECE_CHARS = {1: "P", 2: "N", 3: "B", 4: "R", 5: "Q", 6: "K"}
PROMO_LETTER = {2: "n", 3: "b", 4: "r", 5: "q"}


def sq_name(s):
    return FILES[s & 7] + str((s >> 4) + 1)


def mv_from(m): return m & 127
def mv_to(m): return (m >> 7) & 127
def mv_promo(m): return (m >> 14) & 7
def mv_flag(m): return (m >> 17) & 7


def move_to_uci(m):
    s = sq_name(mv_from(m)) + sq_name(mv_to(m))
    if mv_promo(m):
        s += PROMO_LETTER[mv_promo(m)]
    return s


class Position:
    """0x88 board with make / unmake, incremental hash and evaluation."""

    def __init__(self, fen=START_FEN):
        self.set_fen(fen)

    # ---------------------------------------------------------------- setup
    def set_fen(self, fen):
        self.board = [0] * 128
        self.mg = self.eg = self.phase = 0
        self.hash = 0
        self.king = [4, 116]
        parts = fen.split()
        r, f = 7, 0
        for ch in parts[0]:
            if ch == "/":
                r -= 1
                f = 0
            elif ch.isdigit():
                f += int(ch)
            else:
                t = "pnbrqk".index(ch.lower()) + 1
                p = t | (8 if ch.islower() else 0)
                sq = r * 16 + f
                self._put(sq, p)
                if t == 6:
                    self.king[p >> 3] = sq
                f += 1
        self.side = 0 if (len(parts) < 2 or parts[1] == "w") else 1
        self.castling = 0
        if len(parts) > 2:
            for ch in parts[2]:
                self.castling |= {"K": 1, "Q": 2, "k": 4, "q": 8}.get(ch, 0)
        self.ep = -1
        if len(parts) > 3 and parts[3] != "-":
            self.ep = (int(parts[3][1]) - 1) * 16 + FILES.index(parts[3][0])
        self.halfmove = int(parts[4]) if len(parts) > 4 else 0
        self.fullmove = int(parts[5]) if len(parts) > 5 else 1
        if self.side:
            self.hash ^= ZSIDE
        self.hash ^= ZCASTLE[self.castling]
        if self.ep >= 0:
            self.hash ^= ZEP[self.ep & 7]
        self.stack = []
        self.hist = [self.hash]

    def copy(self):
        p = Position.__new__(Position)
        p.board = self.board[:]
        p.side, p.castling, p.ep = self.side, self.castling, self.ep
        p.halfmove, p.fullmove = self.halfmove, self.fullmove
        p.king = self.king[:]
        p.hash, p.mg, p.eg, p.phase = self.hash, self.mg, self.eg, self.phase
        p.stack = []
        p.hist = self.hist[:]
        return p

    def _put(self, sq, p):
        self.board[sq] = p
        self.hash ^= ZP[p][sq]
        self.mg += PST_MG[p][sq]
        self.eg += PST_EG[p][sq]
        self.phase += PHASE[p & 7]

    def _del(self, sq):
        p = self.board[sq]
        self.board[sq] = 0
        self.hash ^= ZP[p][sq]
        self.mg -= PST_MG[p][sq]
        self.eg -= PST_EG[p][sq]
        self.phase -= PHASE[p & 7]

    # ------------------------------------------------------------ evaluation
    def evaluate(self):
        ph = self.phase if self.phase < 24 else 24
        s = (self.mg * ph + self.eg * (24 - ph)) // 24
        return (s if self.side == 0 else -s) + 8

    # --------------------------------------------------------------- attacks
    def attacked(self, sq, by):
        b = self.board
        if by == 0:
            s = sq - 15
            if not s & 0x88 and b[s] == 1: return True
            s = sq - 17
            if not s & 0x88 and b[s] == 1: return True
        else:
            s = sq + 15
            if not s & 0x88 and b[s] == 9: return True
            s = sq + 17
            if not s & 0x88 and b[s] == 9: return True
        kn = 2 | (by << 3)
        for d in KN_D:
            s = sq + d
            if not s & 0x88 and b[s] == kn: return True
        kg = 6 | (by << 3)
        for d in K_D:
            s = sq + d
            if not s & 0x88 and b[s] == kg: return True
        bi, ro, qu = 3 | (by << 3), 4 | (by << 3), 5 | (by << 3)
        for d in B_D:
            s = sq + d
            while not s & 0x88:
                p = b[s]
                if p:
                    if p == bi or p == qu: return True
                    break
                s += d
        for d in R_D:
            s = sq + d
            while not s & 0x88:
                p = b[s]
                if p:
                    if p == ro or p == qu: return True
                    break
                s += d
        return False

    def in_check(self):
        return self.attacked(self.king[self.side], self.side ^ 1)

    # ------------------------------------------------------------ generation
    def gen(self, caps_only=False):
        b = self.board
        side = self.side
        out = []
        add = out.append
        for sq in SQ64:
            p = b[sq]
            if not p or (p >> 3) != side:
                continue
            t = p & 7
            if t == 1:
                if side == 0:
                    fwd, start, last = 16, 1, 7
                else:
                    fwd, start, last = -16, 6, 0
                to = sq + fwd
                prs = (5,) if caps_only else (5, 4, 3, 2)
                if not b[to]:
                    if (to >> 4) == last:
                        for pr in prs:
                            add(sq | (to << 7) | (pr << 14))
                    elif not caps_only:
                        add(sq | (to << 7))
                        if (sq >> 4) == start and not b[to + fwd]:
                            add(sq | ((to + fwd) << 7) | (F_DOUBLE << 17))
                for d in (fwd - 1, fwd + 1):
                    to = sq + d
                    if to & 0x88:
                        continue
                    tp = b[to]
                    if tp:
                        if (tp >> 3) != side:
                            if (to >> 4) == last:
                                for pr in prs:
                                    add(sq | (to << 7) | (pr << 14))
                            else:
                                add(sq | (to << 7))
                    elif to == self.ep:
                        add(sq | (to << 7) | (F_EP << 17))
            elif t == 2:
                for d in KN_D:
                    to = sq + d
                    if to & 0x88:
                        continue
                    tp = b[to]
                    if tp == 0:
                        if not caps_only:
                            add(sq | (to << 7))
                    elif (tp >> 3) != side:
                        add(sq | (to << 7))
            elif t == 6:
                for d in K_D:
                    to = sq + d
                    if to & 0x88:
                        continue
                    tp = b[to]
                    if tp == 0:
                        if not caps_only:
                            add(sq | (to << 7))
                    elif (tp >> 3) != side:
                        add(sq | (to << 7))
                if not caps_only:
                    c = self.castling
                    if side == 0 and sq == 4 and c & 3:
                        if (c & 1 and not b[5] and not b[6]
                                and not self.attacked(4, 1) and not self.attacked(5, 1) and not self.attacked(6, 1)):
                            add(4 | (6 << 7) | (F_CASTLE << 17))
                        if (c & 2 and not b[3] and not b[2] and not b[1]
                                and not self.attacked(4, 1) and not self.attacked(3, 1) and not self.attacked(2, 1)):
                            add(4 | (2 << 7) | (F_CASTLE << 17))
                    elif side == 1 and sq == 116 and c & 12:
                        if (c & 4 and not b[117] and not b[118]
                                and not self.attacked(116, 0) and not self.attacked(117, 0) and not self.attacked(118, 0)):
                            add(116 | (118 << 7) | (F_CASTLE << 17))
                        if (c & 8 and not b[115] and not b[114] and not b[113]
                                and not self.attacked(116, 0) and not self.attacked(115, 0) and not self.attacked(114, 0)):
                            add(116 | (114 << 7) | (F_CASTLE << 17))
            else:
                dirs = B_D if t == 3 else R_D if t == 4 else K_D
                for d in dirs:
                    to = sq + d
                    while not to & 0x88:
                        tp = b[to]
                        if tp == 0:
                            if not caps_only:
                                add(sq | (to << 7))
                        else:
                            if (tp >> 3) != side:
                                add(sq | (to << 7))
                            break
                        to += d
        return out

    def legal_moves(self):
        out = []
        side = self.side
        for m in self.gen():
            self.make(m)
            if not self.attacked(self.king[side], side ^ 1):
                out.append(m)
            self.unmake()
        return out

    def has_legal(self):
        side = self.side
        for m in self.gen():
            self.make(m)
            ok = not self.attacked(self.king[side], side ^ 1)
            self.unmake()
            if ok:
                return True
        return False

    # ------------------------------------------------------------- make/unmake
    def make(self, m):
        frm = m & 127
        to = (m >> 7) & 127
        promo = (m >> 14) & 7
        flag = (m >> 17) & 7
        b = self.board
        side = self.side
        p = b[frm]
        cap = b[to]
        self.stack.append((m, cap, self.castling, self.ep, self.halfmove,
                           self.hash, self.mg, self.eg, self.phase))
        h = self.hash ^ ZCASTLE[self.castling]
        if self.ep >= 0:
            h ^= ZEP[self.ep & 7]
        self.hash = h
        self.halfmove += 1
        if cap or (p & 7) == 1:
            self.halfmove = 0
        self.ep = -1
        if flag == F_EP:
            self._del(to - 16 if side == 0 else to + 16)
        elif cap:
            self._del(to)
        self._del(frm)
        self._put(to, (promo | (side << 3)) if promo else p)
        if flag == F_DOUBLE:
            opp_pawn = 1 | ((side ^ 1) << 3)
            if b[to - 1] == opp_pawn or b[to + 1] == opp_pawn:
                self.ep = (frm + to) >> 1
        elif flag == F_CASTLE:
            if to > frm:
                rf, rt = frm + 3, frm + 1
            else:
                rf, rt = frm - 4, frm - 1
            rook = b[rf]
            self._del(rf)
            self._put(rt, rook)
        if (p & 7) == 6:
            self.king[side] = to
        self.castling &= CMASK[frm] & CMASK[to]
        self.hash ^= ZCASTLE[self.castling]
        if self.ep >= 0:
            self.hash ^= ZEP[self.ep & 7]
        self.side ^= 1
        self.hash ^= ZSIDE
        if side == 1:
            self.fullmove += 1
        self.hist.append(self.hash)

    def unmake(self):
        m, cap, castling, ep, halfmove, h, mg, eg, ph = self.stack.pop()
        self.hist.pop()
        self.side ^= 1
        if m == -1:  # null move
            self.ep = ep
            self.hash = h
            return
        side = self.side
        frm = m & 127
        to = (m >> 7) & 127
        promo = (m >> 14) & 7
        flag = (m >> 17) & 7
        b = self.board
        p = b[to]
        if promo:
            p = 1 | (side << 3)
        b[frm] = p
        b[to] = cap
        if flag == F_EP:
            b[to - 16 if side == 0 else to + 16] = 1 | ((side ^ 1) << 3)
        elif flag == F_CASTLE:
            if to > frm:
                rf, rt = frm + 3, frm + 1
            else:
                rf, rt = frm - 4, frm - 1
            b[rf] = b[rt]
            b[rt] = 0
        if (p & 7) == 6:
            self.king[side] = frm
        self.castling, self.ep, self.halfmove = castling, ep, halfmove
        self.hash, self.mg, self.eg, self.phase = h, mg, eg, ph
        if side == 1:
            self.fullmove -= 1

    def make_null(self):
        self.stack.append((-1, 0, self.castling, self.ep, self.halfmove,
                           self.hash, 0, 0, 0))
        if self.ep >= 0:
            self.hash ^= ZEP[self.ep & 7]
        self.ep = -1
        self.side ^= 1
        self.hash ^= ZSIDE
        self.hist.append(self.hash)

    # ------------------------------------------------------------- game state
    def insufficient(self):
        minors = 0
        for sq in SQ64:
            p = self.board[sq]
            if p:
                t = p & 7
                if t in (1, 4, 5):
                    return False
                if t in (2, 3):
                    minors += 1
        return minors <= 1


def parse_uci_move(pos, s):
    for m in pos.legal_moves():
        if move_to_uci(m) == s:
            return m
    return None


def san_of(pos, m):
    """Standard algebraic notation of legal move m in pos (before it is played)."""
    frm, to, promo, flag = mv_from(m), mv_to(m), mv_promo(m), mv_flag(m)
    p = pos.board[frm]
    t = p & 7
    if flag == F_CASTLE:
        s = "O-O" if to > frm else "O-O-O"
    else:
        cap = pos.board[to] != 0 or flag == F_EP
        if t == 1:
            s = (FILES[frm & 7] if cap else "") + ("x" if cap else "") + sq_name(to)
            if promo:
                s += "=" + PIECE_CHARS[promo]
        else:
            s = PIECE_CHARS[t]
            others = [x for x in pos.legal_moves()
                      if x != m and mv_to(x) == to and pos.board[mv_from(x)] == p]
            if others:
                same_file = any((mv_from(x) & 7) == (frm & 7) for x in others)
                same_rank = any((mv_from(x) >> 4) == (frm >> 4) for x in others)
                if not same_file:
                    s += FILES[frm & 7]
                elif not same_rank:
                    s += str((frm >> 4) + 1)
                else:
                    s += sq_name(frm)
            if cap:
                s += "x"
            s += sq_name(to)
    pos.make(m)
    if pos.in_check():
        s += "#" if not pos.has_legal() else "+"
    pos.unmake()
    return s


def perft(pos, depth):
    if depth == 0:
        return 1
    n = 0
    side = pos.side
    for m in pos.gen():
        pos.make(m)
        if not pos.attacked(pos.king[side], side ^ 1):
            n += 1 if depth == 1 else perft(pos, depth - 1)
        pos.unmake()
    return n


class Timeout(Exception):
    pass


class Searcher:
    """Iterative-deepening PVS alpha-beta search."""

    def __init__(self):
        self.tt = {}
        self.killers = [[0, 0] for _ in range(130)]
        self.hist = [[0] * 128 for _ in range(16)]
        self.stop = False
        self.nodes = 0
        self.deadline = 0.0
        self.can_stop = False
        self.noise = 0
        self.pos = None
        self.ext_limit = 8
        self.yield_gui = False

    def check_time(self):
        if self.stop or (self.can_stop and time.time() > self.deadline):
            raise Timeout()
        if self.yield_gui:
            time.sleep(0.0004)

    def order(self, moves, tt_move, ply):
        b = self.pos.board
        k = self.killers[ply]
        hist = self.hist
        scored = []
        for m in moves:
            frm = m & 127
            to = (m >> 7) & 127
            if m == tt_move:
                s = 10 ** 7
            else:
                tp = b[to]
                if tp:
                    s = 100000 + (tp & 7) * 16 - (b[frm] & 7)
                elif (m >> 17) & 7 == F_EP:
                    s = 100000 + 15
                elif (m >> 14) & 7:
                    s = 90000 + ((m >> 14) & 7)
                elif m == k[0]:
                    s = 80000
                elif m == k[1]:
                    s = 79000
                else:
                    s = hist[b[frm]][to]
            scored.append((s, m))
        scored.sort(reverse=True)
        return [m for _, m in scored]

    def quiesce(self, alpha, beta, ply):
        self.nodes += 1
        if (self.nodes & 2047) == 0:
            self.check_time()
        pos = self.pos
        stand = pos.evaluate()
        if ply >= 100:
            return stand
        if stand >= beta:
            return beta
        if stand > alpha:
            alpha = stand
        side = pos.side
        b = pos.board
        for m in self.order(pos.gen(True), 0, ply if ply < 128 else 128):
            tp = b[(m >> 7) & 127] & 7
            if stand + MAT[tp] + 200 < alpha and not (m >> 14) & 7:
                continue
            pos.make(m)
            if pos.attacked(pos.king[side], side ^ 1):
                pos.unmake()
                continue
            score = -self.quiesce(-beta, -alpha, ply + 1)
            pos.unmake()
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha

    def negamax(self, depth, alpha, beta, ply, allow_null=True):
        self.nodes += 1
        if (self.nodes & 1023) == 0:
            self.check_time()
        pos = self.pos
        if ply > 0:
            if pos.halfmove >= 100:
                return 0
            h = pos.hist
            n = len(h)
            key = pos.hash
            end = n - 2 - pos.halfmove
            if end < -1:
                end = -1
            for i in range(n - 3, end, -2):
                if h[i] == key:
                    return 0
        side = pos.side
        in_check = pos.attacked(pos.king[side], side ^ 1)
        if in_check and ply < self.ext_limit:
            depth += 1
        if depth <= 0:
            return self.quiesce(alpha, beta, ply)
        if ply >= 120:
            return pos.evaluate()
        alpha0 = alpha
        key = pos.hash
        tt_move = 0
        e = self.tt.get(key)
        if e is not None:
            ed, ef, es, em = e
            tt_move = em
            if ed >= depth and ply > 0:
                if es > MATE - 200:
                    es -= ply
                elif es < -MATE + 200:
                    es += ply
                if ef == 0:
                    return es
                if ef == 1 and es >= beta:
                    return es
                if ef == 2 and es <= alpha:
                    return es
        # null-move pruning
        if allow_null and not in_check and depth >= 3 and ply > 0 and pos.phase >= 4 and beta < MATE - 200:
            if pos.evaluate() >= beta:
                pos.make_null()
                score = -self.negamax(depth - 3 - (depth >= 6), -beta, -beta + 1, ply + 1, False)
                pos.unmake()
                if score >= beta:
                    return beta
        b = pos.board
        best = -INF
        best_move = 0
        legal = 0
        for m in self.order(pos.gen(), tt_move, ply):
            frm = m & 127
            to = (m >> 7) & 127
            promo = (m >> 14) & 7
            is_cap = b[to] != 0 or ((m >> 17) & 7) == F_EP
            piece = b[frm]
            pos.make(m)
            if pos.attacked(pos.king[side], side ^ 1):
                pos.unmake()
                continue
            legal += 1
            if legal == 1:
                score = -self.negamax(depth - 1, -beta, -alpha, ply + 1)
            else:
                red = 0
                if depth >= 3 and legal > 3 and not in_check and not is_cap and not promo:
                    if not pos.attacked(pos.king[side ^ 1], side):
                        red = 1
                score = -self.negamax(depth - 1 - red, -alpha - 1, -alpha, ply + 1)
                if red and score > alpha:
                    score = -self.negamax(depth - 1, -alpha - 1, -alpha, ply + 1)
                if alpha < score < beta:
                    score = -self.negamax(depth - 1, -beta, -alpha, ply + 1)
            pos.unmake()
            if score > best:
                best = score
                best_move = m
                if score > alpha:
                    alpha = score
                    if alpha >= beta:
                        if not is_cap and not promo:
                            k = self.killers[ply]
                            if k[0] != m:
                                k[1] = k[0]
                                k[0] = m
                            self.hist[piece][to] += depth * depth
                        break
        if legal == 0:
            return -MATE + ply if in_check else 0
        flag = 2 if best <= alpha0 else (1 if best >= beta else 0)
        s = best
        if s > MATE - 200:
            s += ply
        elif s < -MATE + 200:
            s -= ply
        if len(self.tt) > 600000:
            self.tt.clear()
        self.tt[key] = (depth, flag, s, best_move)
        return best

    def root(self, depth, moves):
        pos = self.pos
        self.ext_limit = 2 * depth + 4
        alpha = -INF
        best_move, best_score = moves[0], -INF
        scored = []
        for i, m in enumerate(moves):
            pos.make(m)
            if i == 0 or self.noise:
                score = -self.negamax(depth - 1, -INF, INF, 1)
            else:
                score = -self.negamax(depth - 1, -alpha - 1, -alpha, 1)
                if score > alpha:
                    score = -self.negamax(depth - 1, -INF, -alpha, 1)
            pos.unmake()
            scored.append((score, m))
            if score > best_score:
                best_score, best_move = score, m
                if score > alpha:
                    alpha = score
        if self.noise:
            chosen = max(scored, key=lambda sm: sm[0] + random.gauss(0, self.noise))
            return best_score, chosen[1]
        return best_score, best_move

    def think(self, pos, max_depth=64, tlimit=2.0, noise=0, info=None):
        """Returns (best_move, score_for_side_to_move, depth_reached) or None."""
        self.pos = pos
        self.nodes = 0
        self.noise = noise
        self.start = time.time()
        self.deadline = self.start + tlimit
        base = len(pos.stack)
        moves = pos.legal_moves()
        if not moves:
            return None
        if len(moves) == 1:
            return moves[0], pos.evaluate(), 0
        moves = self.order(moves, 0, 0)
        best, best_score, reached = moves[0], 0, 0
        for depth in range(1, max_depth + 1):
            self.can_stop = depth > 1
            try:
                score, mv = self.root(depth, moves)
            except Timeout:
                while len(pos.stack) > base:
                    pos.unmake()
                break
            best, best_score, reached = mv, score, depth
            moves.remove(mv)
            moves.insert(0, mv)
            if info:
                info(depth, score, self.nodes, time.time() - self.start, mv)
            if abs(score) >= MATE - 100 and not noise:
                break
            if time.time() - self.start > tlimit * 0.55:
                break
        return best, best_score, reached


# =============================================================================
#                                  UCI MODE
# =============================================================================
def uci_main():
    pos = Position()

    def out(s):
        sys.stdout.write(s + "\n")
        sys.stdout.flush()

    for line in sys.stdin:
        parts = line.strip().split()
        if not parts:
            continue
        c = parts[0]
        if c == "uci":
            out("id name Grandmaster Chess (Python)")
            out("id author single-file")
            out("uciok")
        elif c == "isready":
            out("readyok")
        elif c == "ucinewgame":
            pos.set_fen(START_FEN)
        elif c == "position":
            i = 1
            if parts[i] == "startpos":
                pos.set_fen(START_FEN)
                i += 1
            elif parts[i] == "fen":
                pos.set_fen(" ".join(parts[i + 1:i + 7]))
                i += 7
            if i < len(parts) and parts[i] == "moves":
                for s in parts[i + 1:]:
                    m = parse_uci_move(pos, s)
                    if m is None:
                        break
                    pos.make(m)
        elif c == "go":
            g = {}
            k = 1
            while k < len(parts):
                if parts[k] in ("depth", "movetime", "wtime", "btime", "winc", "binc", "movestogo"):
                    g[parts[k]] = int(parts[k + 1])
                    k += 2
                else:
                    k += 1
            depth = g.get("depth", 64)
            if "movetime" in g:
                tl = g["movetime"] / 1000.0
            elif "wtime" in g or "btime" in g:
                left = g.get("wtime" if pos.side == 0 else "btime", 30000)
                inc = g.get("winc" if pos.side == 0 else "binc", 0)
                tl = max(0.05, (left / 30.0 + inc * 0.8) / 1000.0)
            elif "depth" in g:
                tl = 3600.0
            else:
                tl = 5.0

            def info(d, sc, n, el, mv):
                if abs(sc) >= MATE - 200:
                    st = "mate %d" % (((MATE - abs(sc) + 1) // 2) * (1 if sc > 0 else -1))
                else:
                    st = "cp %d" % sc
                out("info depth %d score %s nodes %d time %d nps %d pv %s" %
                    (d, st, n, int(el * 1000), int(n / max(el, 1e-3)), move_to_uci(mv)))

            res = Searcher().think(pos.copy(), depth, tl, 0, info)
            out("bestmove " + (move_to_uci(res[0]) if res else "0000"))
        elif c == "quit":
            break


# =============================================================================
#                              GRAPHICS / GAME
# =============================================================================
LEVELS = [("Beginner", 2, 0.5, 160), ("Casual", 3, 1.0, 45), ("Club", 5, 2.5, 0), ("Master", 24, 6.0, 0)]
THEMES = [("Walnut", (240, 217, 181), (181, 136, 99)),
          ("Emerald", (238, 238, 210), (118, 150, 86)),
          ("Ocean", (222, 227, 230), (125, 150, 166)),
          ("Midnight", (170, 178, 200), (80, 90, 120))]
MODES = [("White", "white"), ("Black", "black"), ("2 Players", "two")]

C_TEXT = (232, 234, 240)
C_MUTED = (146, 152, 170)
C_CARD = (30, 32, 41)
C_CARD2 = (38, 41, 52)
C_BTN = (58, 63, 80)
C_BTN_H = (74, 80, 102)
C_ACC = (104, 160, 72)
C_ACC_H = (120, 178, 86)


# ----------------------------------------------------------- piece rendering
def _chaikin(pts, it):
    for _ in range(it):
        new = []
        n = len(pts)
        for i in range(n):
            a, b = pts[i], pts[(i + 1) % n]
            new.append((0.75 * a[0] + 0.25 * b[0], 0.75 * a[1] + 0.25 * b[1]))
            new.append((0.25 * a[0] + 0.75 * b[0], 0.25 * a[1] + 0.75 * b[1]))
        pts = new
    return pts


def _piece_shapes(t):
    base = [("rrect", 19, 83, 62, 9, 4), ("rrect", 26, 75, 48, 10, 4)]
    if t == 1:
        return ([("poly", [(44, 46), (56, 46), (60, 58), (68, 76), (32, 76), (40, 58)], 2)] + base +
                [("rrect", 36, 43, 28, 7, 3.5), ("circle", 50, 30, 12.5)])
    if t == 4:
        top = [(28, 14), (39, 14), (39, 21), (44, 21), (44, 14), (56, 14), (56, 21), (61, 21),
               (61, 14), (72, 14), (72, 36), (28, 36)]
        return ([("poly", [(35, 36), (65, 36), (62, 70), (68, 76), (32, 76), (38, 70)], 0)] + base +
                [("poly", top, 0), ("rrect", 29, 35, 42, 7, 2.5)])
    if t == 3:
        return ([("poly", [(43, 44), (57, 44), (62, 60), (68, 76), (32, 76), (38, 60)], 2)] + base +
                [("rrect", 37, 47, 26, 7, 3.5), ("ellipse", 50, 32, 12.5, 17), ("circle", 50, 12, 4.6)])
    if t == 2:
        pts = [(30, 76), (32, 64), (38, 54), (42, 48), (33, 50), (24, 52), (20, 47), (22, 41), (30, 34),
               (35, 28), (36, 17), (44, 23), (52, 23), (63, 30), (72, 44), (76, 60), (74, 76)]
        return [("poly", pts, 1)] + base
    if t == 5:
        crown = [(24, 28), (30, 42), (37, 22), (43, 42), (50, 18), (57, 42), (63, 22), (70, 42), (76, 28),
                 (67, 70), (70, 76), (30, 76), (33, 70)]
        return ([("poly", crown, 1)] + base +
                [("rrect", 31, 63, 38, 9, 4), ("circle", 24, 26, 4.8), ("circle", 37, 20, 4.8),
                 ("circle", 50, 16, 4.8), ("circle", 63, 20, 4.8), ("circle", 76, 26, 4.8)])
    body = [(36, 34), (64, 34), (71, 46), (65, 56), (68, 70), (72, 76), (28, 76), (32, 70), (35, 56), (29, 46)]
    return ([("poly", body, 2)] + base +
            [("rrect", 30, 63, 40, 9, 4), ("rrect", 34, 29, 32, 8, 3.5),
             ("rrect", 46, 4, 8, 27, 2.2), ("rrect", 39, 11, 22, 8, 2.2)])


def _draw_shape(surf, sh, fill, outline, sc, ow):
    k = sh[0]
    if k == "poly":
        pts = sh[1]
        if sh[2]:
            pts = _chaikin(pts, sh[2])
        P = [(x * sc, y * sc) for x, y in pts]
        pygame.draw.polygon(surf, fill, P)
        pygame.draw.polygon(surf, outline, P, ow)
        for q in P:
            pygame.draw.circle(surf, outline, (int(q[0]), int(q[1])), ow // 2)
    elif k == "rrect":
        r = pygame.Rect(int(sh[1] * sc), int(sh[2] * sc), int(sh[3] * sc), int(sh[4] * sc))
        pygame.draw.rect(surf, fill, r, border_radius=int(sh[5] * sc))
        pygame.draw.rect(surf, outline, r, ow, border_radius=int(sh[5] * sc))
    elif k == "circle":
        c = (int(sh[1] * sc), int(sh[2] * sc))
        pygame.draw.circle(surf, fill, c, int(sh[3] * sc))
        pygame.draw.circle(surf, outline, c, int(sh[3] * sc), ow)
    elif k == "ellipse":
        r = pygame.Rect(int((sh[1] - sh[3]) * sc), int((sh[2] - sh[4]) * sc), int(2 * sh[3] * sc), int(2 * sh[4] * sc))
        pygame.draw.ellipse(surf, fill, r)
        pygame.draw.ellipse(surf, outline, r, ow)


def _draw_details(surf, t, color, sc, ow):
    def line(pts, w):
        P = [(int(x * sc), int(y * sc)) for x, y in pts]
        pygame.draw.lines(surf, color, False, P, w)
        for q in (P[0], P[-1]):
            pygame.draw.circle(surf, color, q, w // 2)

    if t == 3:
        line([(43, 38), (57, 24)], int(ow * 1.1))
    elif t == 2:
        pygame.draw.circle(surf, color, (int(40 * sc), int(31 * sc)), int(2.8 * sc))
        pygame.draw.circle(surf, color, (int(23.5 * sc), int(45 * sc)), int(1.8 * sc))
        line([(53, 30), (61, 37), (64, 49)], int(ow * 0.8))
        line([(59, 29), (68, 38), (71, 52)], int(ow * 0.8))
    elif t == 1:
        pass


def build_piece_masters():
    S = 400
    sc = S / 100.0
    masters = {}
    for color in (0, 1):
        if color == 0:
            fill, outline, detail, dark = (252, 248, 238), (54, 45, 40), (54, 45, 40), 0.36
        else:
            fill, outline, detail, dark = (104, 102, 118), (10, 10, 14), (212, 214, 226), 0.58
        ow = max(2, int(2.9 * sc))
        grad = pygame.Surface((S, S))
        for x in range(S):
            u = (x / S - 0.26) / 0.56
            u = 0 if u < 0 else 1 if u > 1 else u
            u = u * u * (3 - 2 * u)
            v = int(255 * (1 - dark * u))
            pygame.draw.line(grad, (v, v, v), (x, 0), (x, S))
        for t in range(1, 7):
            canvas = pygame.Surface((S, S), pygame.SRCALPHA)
            for sh in _piece_shapes(t):
                _draw_shape(canvas, sh, fill, outline, sc, ow)
            canvas.blit(grad, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
            _draw_details(canvas, t, detail, sc, ow)
            masters[t | (color << 3)] = canvas
    return masters


def make_sound(freqs, dur, vol, noise=0.0, decay=18.0):
    sr = 44100
    n = int(sr * dur)
    buf = array.array("h")
    for i in range(n):
        t = i / sr
        env = math.exp(-decay * t) * min(1.0, i / (sr * 0.002))
        v = 0.0
        for f, a in freqs:
            v += a * math.sin(2 * math.pi * f * t)
        if noise:
            v += noise * (random.random() * 2 - 1)
        buf.append(int(max(-1.0, min(1.0, v * env * vol)) * 32767))
    return pygame.mixer.Sound(buffer=buf.tobytes())


# ------------------------------------------------------------------- the app
class App:
    MIN_W, MIN_H = 900, 640

    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Grandmaster Chess")
        self.screen = pygame.display.set_mode((1120, 740), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        names = "segoeui,helveticaneue,helvetica,arial,dejavusans,liberationsans,freesans"
        mk = lambda size, bold=False: pygame.font.SysFont(names, size, bold=bold)
        self.f_tiny, self.f_small, self.f_body = mk(12, True), mk(14), mk(15)
        self.f_bold, self.f_head, self.f_big = mk(16, True), mk(24, True), mk(46, True)
        self.f_mid = mk(22, True)
        self.masters = build_piece_masters()
        self.sounds = {}
        self.sound_on = True
        self.init_sound()
        self.theme, self.level, self.mode_i = 0, 2, 0
        self.pcache, self.tcache = {}, {}
        self.cur_searcher = None
        self.job_id = 0
        self.ai_busy = False
        self.busy_kind = None
        self.ai_result = None
        self.ai_min_time = 0
        self.pressed = None
        self.toast = None
        self.cursor_hand = False
        self.eval_disp = 0.5
        self.W = self.H = 0
        self._bs_key = None
        self.layout()
        self.new_game()

    # ------------------------------------------------------------------ misc
    @property
    def mode(self):
        return MODES[self.mode_i][1]

    def init_sound(self):
        try:
            pygame.mixer.init(44100, -16, 1, 512)
            self.sounds = {
                "move": make_sound([(170, .7), (85, .6)], .10, .8, .22, 38),
                "capture": make_sound([(140, .8), (290, .35)], .15, .9, .55, 28),
                "check": make_sound([(660, .5), (990, .3)], .28, .55, .0, 11),
                "end": make_sound([(392, .4), (494, .4), (587, .4), (784, .3)], .75, .6, .0, 4.5),
            }
        except Exception:
            self.sounds = {}

    def play_sound(self, name):
        if self.sound_on and name in self.sounds:
            try:
                self.sounds[name].play()
            except Exception:
                pass

    def show_toast(self, text, secs=2.6):
        self.toast = (text, time.time() + secs)

    def txt(self, s, font, color):
        key = (s, id(font), color)
        surf = self.tcache.get(key)
        if surf is None:
            if len(self.tcache) > 1500:
                self.tcache.clear()
            surf = font.render(s, True, color)
            self.tcache[key] = surf
        return surf

    def blit_text(self, s, font, color, x, y, anchor="topleft"):
        surf = self.txt(s, font, color)
        r = surf.get_rect()
        setattr(r, anchor, (x, y))
        self.screen.blit(surf, r)
        return r

    # ---------------------------------------------------------------- layout
    def layout(self):
        W, H = self.screen.get_size()
        self.W, self.H = W, H
        mar, self.bar_w, gap, pgap, pw = 24, 26, 10, 28, 340
        avail = W - 2 * mar - self.bar_w - gap - pgap - pw
        sq = max(40, min(avail // 8, (H - 2 * mar) // 8))
        self.sq = sq
        board = 8 * sq
        content = self.bar_w + gap + board + pgap + pw
        x0, y0 = max(0, (W - content) // 2), (H - board) // 2
        self.bar_rect = pygame.Rect(x0, y0, self.bar_w, board)
        self.bx, self.by = x0 + self.bar_w + gap, y0
        self.panel = pygame.Rect(self.bx + board + pgap, y0, pw, board)
        self.pcache.clear()
        self._bs_key = None
        self.f_coord = pygame.font.SysFont("segoeui,helveticaneue,helvetica,arial,dejavusans", max(11, int(sq * 0.17)), bold=True)
        # background
        bg = pygame.Surface((W, H))
        for y in range(H):
            t = y / max(1, H - 1)
            pygame.draw.line(bg, (int(30 - 12 * t), int(33 - 13 * t), int(44 - 17 * t)), (0, y), (W, y))
        self.bg = bg
        # overlays
        self.ov_last = pygame.Surface((sq, sq), pygame.SRCALPHA)
        self.ov_last.fill((255, 214, 64, 110))
        self.ov_sel = pygame.Surface((sq, sq), pygame.SRCALPHA)
        self.ov_sel.fill((120, 205, 100, 150))
        self.ov_hover = pygame.Surface((sq, sq), pygame.SRCALPHA)
        self.ov_hover.fill((255, 255, 255, 60))
        self.ov_dot = pygame.Surface((sq, sq), pygame.SRCALPHA)
        pygame.draw.circle(self.ov_dot, (20, 20, 20, 78), (sq // 2, sq // 2), int(sq * 0.17))
        self.ov_ring = pygame.Surface((sq, sq), pygame.SRCALPHA)
        pygame.draw.circle(self.ov_ring, (20, 20, 20, 78), (sq // 2, sq // 2), int(sq * 0.47), max(3, int(sq * 0.085)))
        self.ov_glow = pygame.Surface((sq, sq), pygame.SRCALPHA)
        c = sq / 2
        for i in range(int(c), 0, -1):
            tt = 1 - i / c
            pygame.draw.circle(self.ov_glow, (235, 45, 45, int(215 * tt ** 1.5)), (int(c), int(c)), i)
        self.ov_shadow = pygame.Surface((sq, sq), pygame.SRCALPHA)
        pygame.draw.ellipse(self.ov_shadow, (0, 0, 0, 38), pygame.Rect(int(sq * .16), int(sq * .74), int(sq * .68), int(sq * .17)))
        pygame.draw.ellipse(self.ov_shadow, (0, 0, 0, 52), pygame.Rect(int(sq * .22), int(sq * .77), int(sq * .56), int(sq * .12)))
        self.bar_mask = pygame.Surface((self.bar_w, board), pygame.SRCALPHA)
        pygame.draw.rect(self.bar_mask, (255, 255, 255, 255), (0, 0, self.bar_w, board), border_radius=8)
        # panel rects
        p = self.panel
        y = p.top
        self.r_top = pygame.Rect(p.x, y, p.w, 58)
        y += 58 + 10
        self.r_status = pygame.Rect(p.x, y, p.w, 34)
        y += 34 + 10
        bt_h, bgap, rows = 38, 8, 4
        area = rows * bt_h + (rows - 1) * bgap
        self.r_bottom = pygame.Rect(p.x, p.bottom - area - 10 - 58, p.w, 58)
        self.r_moves = pygame.Rect(p.x, y, p.w, self.r_bottom.top - 10 - y)
        self.build_buttons(p.bottom - area, bt_h, bgap)

    def build_buttons(self, y0, h, gap):
        p = self.panel
        w2 = (p.w - gap) // 2
        w3 = (p.w - 2 * gap) // 3
        human_turn = lambda: self.human_can_move()
        B = []

        def add(x, y, w, label, action, enabled=lambda: True, accent=False):
            B.append(dict(rect=pygame.Rect(x, y, w, h), label=label, action=action, enabled=enabled, accent=accent))

        y = y0
        add(p.x, y, w2, lambda: "New Game", self.new_game, accent=True)
        add(p.x + w2 + gap, y, w2, lambda: "Undo", self.undo, lambda: bool(self.moves))
        y += h + gap
        add(p.x, y, w3, lambda: "Hint", self.hint_request, lambda: human_turn() and not self.promo)
        add(p.x + w3 + gap, y, w3, lambda: "Flip", self.flip)
        add(p.x + 2 * (w3 + gap), y, w3, lambda: "Sound " + ("On" if self.sound_on else "Off"), self.toggle_sound)
        y += h + gap
        add(p.x, y, w2, lambda: "Play as: " + MODES[self.mode_i][0], self.cycle_mode)
        add(p.x + w2 + gap, y, w2, lambda: "Level: " + LEVELS[self.level][0], self.cycle_level)
        y += h + gap
        add(p.x, y, w2, lambda: "Theme: " + THEMES[self.theme][0], self.cycle_theme)
        add(p.x + w2 + gap, y, w2, lambda: "Save PGN", self.save_pgn, lambda: bool(self.moves))
        self.buttons = B

    # ----------------------------------------------------------- game control
    def new_game(self):
        self.cancel_search()
        self.pos = Position()
        self.moves, self.sans = [], []
        self.last = None
        self.sel, self.sel_targets, self.sel_caps = None, set(), set()
        self.dragging, self.drag_start, self.drag_pos = False, None, (0, 0)
        self.anim = None
        self.hint = None
        self.promo = None
        self.show_result = True
        self.scroll, self.autoscroll = 0, True
        self.flipped = self.mode == "black"
        self.refresh()
        self.set_eval_static()
        self.maybe_start_ai()

    def refresh(self):
        self.legal = self.pos.legal_moves()
        self.in_check = self.pos.in_check()
        self.result = self.compute_result()

    def compute_result(self):
        pos = self.pos
        if not self.legal:
            if self.in_check:
                return ("Checkmate", "Black wins" if pos.side == 0 else "White wins", "0-1" if pos.side == 0 else "1-0")
            return ("Stalemate", "Draw", "1/2-1/2")
        if pos.halfmove >= 100:
            return ("Draw", "Fifty-move rule", "1/2-1/2")
        if pos.hist.count(pos.hash) >= 3:
            return ("Draw", "Threefold repetition", "1/2-1/2")
        if pos.insufficient():
            return ("Draw", "Insufficient material", "1/2-1/2")
        return None

    def human_color(self):
        return None if self.mode == "two" else (0 if self.mode == "white" else 1)

    def is_ai_turn(self):
        h = self.human_color()
        return h is not None and self.pos.side != h

    def human_can_move(self):
        return not self.result and not self.is_ai_turn() and not self.ai_busy

    def set_eval(self, cp):
        if self.result:
            r = self.result[2]
            self.eval_label = r
            self.eval_target = 1.0 if r == "1-0" else 0.0 if r == "0-1" else 0.5
            return
        if abs(cp) >= MATE - 200:
            self.eval_label = "M%d" % ((MATE - abs(cp) + 1) // 2)
            self.eval_target = 0.98 if cp > 0 else 0.02
        else:
            self.eval_label = "%.1f" % (abs(cp) / 100.0)
            self.eval_target = 1 / (1 + 10 ** (-cp / 400.0))
        self.eval_cp = cp

    def set_eval_static(self):
        v = self.pos.evaluate() - 8
        self.set_eval(v if self.pos.side == 0 else -v)

    # ------------------------------------------------------------ playing a move
    def play(self, m, animate=True):
        pos = self.pos
        san = san_of(pos, m)
        frm, to, flag = mv_from(m), mv_to(m), mv_flag(m)
        side = pos.side
        cap_piece = pos.board[to]
        cap_sq = to
        if flag == F_EP:
            cap_sq = to - 16 if side == 0 else to + 16
            cap_piece = pos.board[cap_sq]
        pos.make(m)
        self.moves.append(m)
        self.sans.append(san)
        self.last = (frm, to)
        items = [(pos.board[to], frm, to)]
        if flag == F_CASTLE:
            rf, rt = (to + 1, to - 1) if to > frm else (to - 2, to + 1)
            items.append((pos.board[rt], rf, rt))
        self.anim = None
        if animate:
            self.anim = dict(items=items, cap=(cap_piece, cap_sq) if cap_piece else None,
                             t0=time.time(), dur=0.30 if flag == F_CASTLE else 0.22)
        self.sel, self.sel_targets, self.sel_caps = None, set(), set()
        self.hint = None
        self.promo = None
        self.autoscroll = True
        self.show_result = True
        self.refresh()
        if self.result:
            self.play_sound("end")
        elif self.in_check:
            self.play_sound("check")
        elif cap_piece:
            self.play_sound("capture")
        else:
            self.play_sound("move")
        if not self.is_ai_turn() or self.result:
            self.set_eval_static()
        self.maybe_start_ai()

    def try_move(self, frm, to, animate):
        cands = [m for m in self.legal if mv_from(m) == frm and mv_to(m) == to]
        if not cands:
            return False
        if len(cands) > 1:
            self.promo = (frm, to, cands)
            self.sel = frm
            return True
        self.play(cands[0], animate)
        return True

    def select(self, s):
        self.sel = s
        self.sel_targets, self.sel_caps = set(), set()
        for m in self.legal:
            if mv_from(m) == s:
                t = mv_to(m)
                self.sel_targets.add(t)
                if self.pos.board[t] or mv_flag(m) == F_EP:
                    self.sel_caps.add(t)

    def clear_sel(self):
        self.sel, self.sel_targets, self.sel_caps = None, set(), set()

    def undo(self):
        if not self.moves:
            return
        self.cancel_search()
        self.anim = None
        self.promo = None
        self.pop_move()
        h = self.human_color()
        if h is not None:
            while self.moves and self.pos.side != h:
                self.pop_move()
        self.last = (mv_from(self.moves[-1]), mv_to(self.moves[-1])) if self.moves else None
        self.clear_sel()
        self.hint = None
        self.show_result = True
        self.autoscroll = True
        self.refresh()
        self.set_eval_static()
        self.maybe_start_ai()

    def pop_move(self):
        self.pos.unmake()
        self.moves.pop()
        self.sans.pop()

    def flip(self):
        self.flipped = not self.flipped
        self._bs_key = None

    def toggle_sound(self):
        self.sound_on = not self.sound_on

    def cycle_mode(self):
        self.mode_i = (self.mode_i + 1) % len(MODES)
        self.new_game()
        self.show_toast("New game started")

    def cycle_level(self):
        self.level = (self.level + 1) % len(LEVELS)
        self.show_toast("Computer level: " + LEVELS[self.level][0])

    def cycle_theme(self):
        self.theme = (self.theme + 1) % len(THEMES)
        self._bs_key = None

    def save_pgn(self):
        if not self.moves:
            return
        res = self.result[2] if self.result else "*"
        h = self.human_color()
        eng = "Grandmaster Chess (%s)" % LEVELS[self.level][0]
        white = "Player" if h in (0, None) else eng
        black = "Player" if h in (1, None) else eng
        if h is None:
            white, black = "Player 1", "Player 2"
        tokens = []
        for i, s in enumerate(self.sans):
            if i % 2 == 0:
                tokens.append("%d." % (i // 2 + 1))
            tokens.append(s)
        tokens.append(res)
        lines, cur = [], ""
        for tkn in tokens:
            if len(cur) + len(tkn) + 1 > 80:
                lines.append(cur)
                cur = tkn
            else:
                cur = (cur + " " + tkn).strip()
        lines.append(cur)
        now = datetime.datetime.now()
        text = ('[Event "Casual Game"]\n[Site "Grandmaster Chess"]\n[Date "%s"]\n[Round "-"]\n'
                '[White "%s"]\n[Black "%s"]\n[Result "%s"]\n\n%s\n') % (
            now.strftime("%Y.%m.%d"), white, black, res, "\n".join(lines))
        path = os.path.join(os.getcwd(), "chess_%s.pgn" % now.strftime("%Y%m%d_%H%M%S"))
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            self.show_toast("Saved " + os.path.basename(path), 3.5)
        except Exception as ex:
            self.show_toast("Could not save: %s" % ex, 3.5)

    # ------------------------------------------------------------------ AI
    def maybe_start_ai(self):
        if self.result or self.ai_busy or not self.is_ai_turn():
            return
        self.start_search("move")

    def hint_request(self):
        if self.human_can_move():
            self.start_search("hint")

    def start_search(self, kind):
        self.job_id += 1
        job = self.job_id
        s = Searcher()
        s.yield_gui = True
        self.cur_searcher = s
        posc = self.pos.copy()
        if kind == "move":
            _, depth, tl, noise = LEVELS[self.level]
            self.ai_min_time = time.time() + 0.45
        else:
            depth, tl, noise = 6, 2.0, 0
            self.ai_min_time = 0
        self.ai_busy = True
        self.busy_kind = kind
        self.ai_result = None

        def run():
            try:
                res = s.think(posc, depth, tl, noise)
            except Exception:
                res = None
            if not s.stop:
                self.ai_result = (job, kind, res)

        threading.Thread(target=run, daemon=True).start()

    def cancel_search(self):
        if self.cur_searcher:
            self.cur_searcher.stop = True
        self.job_id += 1
        self.ai_busy = False
        self.busy_kind = None
        self.ai_result = None

    # -------------------------------------------------------------- update
    def update(self):
        now = time.time()
        if self.anim and now >= self.anim["t0"] + self.anim["dur"]:
            self.anim = None
        r = self.ai_result
        if r and r[0] == self.job_id and not self.anim and now >= self.ai_min_time:
            self.ai_result = None
            self.ai_busy = False
            self.busy_kind = None
            _, kind, res = r
            if res:
                m, score, _d = res
                if kind == "move" and m in self.legal and not self.result:
                    cp = score if self.pos.side == 0 else -score
                    self.play(m, True)
                    if not self.result:
                        self.set_eval(cp)
                elif kind == "hint" and m in self.legal:
                    self.hint = (mv_from(m), mv_to(m))
        self.eval_disp += (self.eval_target - self.eval_disp) * 0.14
        if self.toast and now > self.toast[1]:
            self.toast = None

    # ------------------------------------------------------------ geometry
    def sq_xy(self, s):
        f, r = s & 7, s >> 4
        if not self.flipped:
            return self.bx + f * self.sq, self.by + (7 - r) * self.sq
        return self.bx + (7 - f) * self.sq, self.by + r * self.sq

    def sq_at(self, pos):
        x, y = pos[0] - self.bx, pos[1] - self.by
        if x < 0 or y < 0 or x >= 8 * self.sq or y >= 8 * self.sq:
            return None
        col, row = x // self.sq, y // self.sq
        if not self.flipped:
            return (7 - row) * 16 + col
        return row * 16 + (7 - col)

    # --------------------------------------------------------------- events
    def handle(self, ev):
        if ev.type == pygame.QUIT:
            return False
        if ev.type == pygame.KEYDOWN:
            self.on_key(ev)
        elif ev.type == pygame.MOUSEWHEEL:
            if self.r_moves.collidepoint(pygame.mouse.get_pos()):
                self.scroll -= ev.y * 28
        elif ev.type == pygame.MOUSEBUTTONDOWN:
            if ev.button == 1:
                self.mouse_down(ev.pos)
            elif ev.button == 3:
                self.clear_sel()
                self.dragging = False
        elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
            self.mouse_up(ev.pos)
        elif ev.type == pygame.MOUSEMOTION:
            self.drag_pos = ev.pos
        return True

    def on_key(self, ev):
        k = ev.key
        mods = pygame.key.get_mods()
        if k == pygame.K_ESCAPE:
            if self.promo:
                self.promo = None
                self.clear_sel()
            elif self.sel is not None:
                self.clear_sel()
            elif self.result:
                self.show_result = False
        elif k == pygame.K_n:
            self.new_game()
        elif k == pygame.K_u or (k == pygame.K_z and mods & pygame.KMOD_CTRL):
            self.undo()
        elif k == pygame.K_f:
            self.flip()
        elif k == pygame.K_h:
            self.hint_request()
        elif k == pygame.K_t:
            self.cycle_theme()
        elif k == pygame.K_m:
            self.toggle_sound()
        elif k == pygame.K_s:
            self.save_pgn()

    def promo_layout(self):
        sq = self.sq
        w, h = int(sq * 4.4), int(sq * 1.7)
        cx, cy = self.bx + 4 * sq, self.by + 4 * sq
        card = pygame.Rect(0, 0, w, h)
        card.center = (cx, cy)
        cells = []
        for i, t in enumerate((5, 4, 3, 2)):
            r = pygame.Rect(card.x + int(sq * 0.2) + i * sq * 1.0, card.y + int(sq * 0.55), sq, sq)
            cells.append((t, r))
        return card, cells

    def overlay_buttons(self):
        bd = 8 * self.sq
        w, h = int(bd * 0.62), int(bd * 0.46)
        card = pygame.Rect(0, 0, w, h)
        card.center = (self.bx + bd // 2, self.by + bd // 2)
        bw = (w - 3 * 16) // 2
        r1 = pygame.Rect(card.x + 16, card.bottom - 62, bw, 44)
        r2 = pygame.Rect(card.x + 32 + bw, card.bottom - 62, bw, 44)
        return card, r1, r2

    def mouse_down(self, pos):
        if self.promo:
            frm, to, cands = self.promo
            card, cells = self.promo_layout()
            for t, r in cells:
                if r.collidepoint(pos):
                    for m in cands:
                        if mv_promo(m) == t:
                            self.play(m, True)
                            return
            if not card.collidepoint(pos):
                self.promo = None
                self.clear_sel()
            return
        if self.result and self.show_result and not self.anim:
            card, r1, r2 = self.overlay_buttons()
            if r1.collidepoint(pos):
                self.new_game()
            elif r2.collidepoint(pos):
                self.show_result = False
            return
        for b in self.buttons:
            if b["rect"].collidepoint(pos) and b["enabled"]():
                self.pressed = b
                return
        if not self.human_can_move() or self.anim:
            return
        s = self.sq_at(pos)
        if s is None:
            self.clear_sel()
            return
        if self.sel is not None and s in self.sel_targets:
            self.try_move(self.sel, s, True)
            return
        p = self.pos.board[s]
        if p and (p >> 3) == self.pos.side:
            self.select(s)
            self.dragging, self.drag_start, self.drag_pos = True, s, pos
        else:
            self.clear_sel()

    def mouse_up(self, pos):
        if self.pressed:
            b = self.pressed
            self.pressed = None
            if b["rect"].collidepoint(pos) and b["enabled"]():
                b["action"]()
            return
        if self.dragging:
            self.dragging = False
            s = self.sq_at(pos)
            if s is not None and s != self.drag_start and s in self.sel_targets:
                self.try_move(self.drag_start, s, False)

    def update_cursor(self):
        mx, my = pygame.mouse.get_pos()
        hand = any(b["rect"].collidepoint(mx, my) and b["enabled"]() for b in self.buttons)
        if not hand and self.human_can_move() and not self.anim and not self.promo:
            s = self.sq_at((mx, my))
            if s is not None:
                p = self.pos.board[s]
                hand = bool(p and (p >> 3) == self.pos.side) or (self.sel is not None and s in self.sel_targets)
        if hand != self.cursor_hand:
            self.cursor_hand = hand
            try:
                pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND if hand else pygame.SYSTEM_CURSOR_ARROW)
            except Exception:
                pass

    # -------------------------------------------------------------- drawing
    def piece_surf(self, p, size):
        key = (p, size)
        s = self.pcache.get(key)
        if s is None:
            s = pygame.transform.smoothscale(self.masters[p], (size, size))
            self.pcache[key] = s
        return s

    def draw_piece(self, p, x, y, shadow=True):
        sq = self.sq
        size = int(sq * 0.94)
        off = (sq - size) // 2
        if shadow:
            self.screen.blit(self.ov_shadow, (x, y))
        self.screen.blit(self.piece_surf(p, size), (x + off, y + off))

    def get_board_surf(self):
        key = (self.theme, self.flipped, self.sq)
        if key == self._bs_key:
            return self._bs
        sq = self.sq
        board = 8 * sq
        pad = 14
        surf = pygame.Surface((board + 2 * pad, board + 2 * pad), pygame.SRCALPHA)
        for i in range(pad, 0, -1):
            a = int(80 * (1 - i / pad) ** 2)
            pygame.draw.rect(surf, (0, 0, 0, a), pygame.Rect(pad - i, pad - i + 3, board + 2 * i, board + 2 * i), border_radius=8 + i)
        sqs = pygame.Surface((board, board), pygame.SRCALPHA)
        _, light, dark = THEMES[self.theme]
        for row in range(8):
            for col in range(8):
                f = col if not self.flipped else 7 - col
                r = 7 - row if not self.flipped else row
                is_light = (r + f) % 2 == 1
                pygame.draw.rect(sqs, light if is_light else dark, (col * sq, row * sq, sq, sq))
                lab = dark if is_light else light
                if col == 0:
                    t = self.f_coord.render(str(r + 1), True, lab)
                    sqs.blit(t, (col * sq + 4, row * sq + 3))
                if row == 7:
                    t = self.f_coord.render(FILES[f], True, lab)
                    sqs.blit(t, ((col + 1) * sq - t.get_width() - 4, (row + 1) * sq - t.get_height() - 2))
        mask = pygame.Surface((board, board), pygame.SRCALPHA)
        pygame.draw.rect(mask, (255, 255, 255, 255), (0, 0, board, board), border_radius=6)
        sqs.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        surf.blit(sqs, (pad, pad))
        self._bs, self._bs_key = surf, key
        return surf

    def draw_eval_bar(self):
        r = self.bar_rect
        tmp = pygame.Surface((r.w, r.h), pygame.SRCALPHA)
        tmp.fill((52, 54, 62))
        wh = int(r.h * self.eval_disp)
        if not self.flipped:
            wr = pygame.Rect(0, r.h - wh, r.w, wh)
        else:
            wr = pygame.Rect(0, 0, r.w, wh)
        pygame.draw.rect(tmp, (238, 238, 232), wr)
        pygame.draw.line(tmp, (120, 124, 138), (0, r.h // 2), (r.w, r.h // 2), 1)
        tmp.blit(self.bar_mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        self.screen.blit(tmp, r)
        white_leads = self.eval_target >= 0.5
        at_bottom = white_leads != self.flipped
        col = (40, 40, 46) if white_leads else (225, 225, 232)
        if at_bottom:
            self.blit_text(self.eval_label, self.f_tiny, col, r.centerx, r.bottom - 6, "midbottom")
        else:
            self.blit_text(self.eval_label, self.f_tiny, col, r.centerx, r.top + 6, "midtop")

    def draw_arrow(self, a, b, color):
        sq = self.sq
        ax, ay = self.sq_xy(a)
        bx, by = self.sq_xy(b)
        ax, ay, bx, by = ax + sq / 2, ay + sq / 2, bx + sq / 2, by + sq / 2
        dx, dy = bx - ax, by - ay
        L = math.hypot(dx, dy)
        if L < 1:
            return
        ux, uy = dx / L, dy / L
        nx, ny = -uy, ux
        sw, hw, hl = sq * 0.14, sq * 0.34, sq * 0.42
        start = (ax + ux * sq * 0.2, ay + uy * sq * 0.2)
        end = (bx - ux * sq * 0.12, by - uy * sq * 0.12)
        neck = (end[0] - ux * hl, end[1] - uy * hl)
        pts = [(start[0] + nx * sw, start[1] + ny * sw), (neck[0] + nx * sw, neck[1] + ny * sw),
               (neck[0] + nx * hw, neck[1] + ny * hw), end,
               (neck[0] - nx * hw, neck[1] - ny * hw), (neck[0] - nx * sw, neck[1] - ny * sw),
               (start[0] - nx * sw, start[1] - ny * sw)]
        layer = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
        pygame.draw.polygon(layer, color, pts)
        self.screen.blit(layer, (0, 0))

    def draw_board(self):
        S = self.screen
        sq = self.sq
        S.blit(self.get_board_surf(), (self.bx - 14, self.by - 14))
        if self.last:
            for s in self.last:
                S.blit(self.ov_last, self.sq_xy(s))
        if self.sel is not None:
            S.blit(self.ov_sel, self.sq_xy(self.sel))
        if self.in_check and not (self.result and self.result[0] != "Checkmate"):
            S.blit(self.ov_glow, self.sq_xy(self.pos.king[self.pos.side]))
        mouse = pygame.mouse.get_pos()
        hs = self.sq_at(mouse)
        for s in self.sel_targets:
            x, y = self.sq_xy(s)
            if s == hs:
                S.blit(self.ov_hover, (x, y))
            S.blit(self.ov_ring if s in self.sel_caps else self.ov_dot, (x, y))
        # pieces
        hidden = set()
        if self.anim:
            for it in self.anim["items"]:
                hidden.add(it[2])
        if self.dragging and self.drag_start is not None:
            hidden.add(self.drag_start)
        b = self.pos.board
        if self.anim and self.anim["cap"]:
            cp, cs = self.anim["cap"]
            x, y = self.sq_xy(cs)
            self.draw_piece(cp, x, y)
        for s in SQ64:
            p = b[s]
            if p and s not in hidden:
                x, y = self.sq_xy(s)
                self.draw_piece(p, x, y)
        if self.hint:
            self.draw_arrow(self.hint[0], self.hint[1], (70, 190, 110, 175))
        if self.anim:
            t = min(1.0, (time.time() - self.anim["t0"]) / self.anim["dur"])
            e = 1 - (1 - t) ** 3
            for p, fs, ts in self.anim["items"]:
                x0, y0 = self.sq_xy(fs)
                x1, y1 = self.sq_xy(ts)
                self.draw_piece(p, x0 + (x1 - x0) * e, y0 + (y1 - y0) * e)
        if self.dragging and self.drag_start is not None:
            p = b[self.drag_start]
            size = int(sq * 1.08)
            surf = self.piece_surf(p, size)
            self.screen.blit(self.ov_shadow, (self.drag_pos[0] - sq // 2 + 6, self.drag_pos[1] - sq // 2 + 8))
            self.screen.blit(surf, (self.drag_pos[0] - size // 2, self.drag_pos[1] - size // 2 - 4))

    def draw_promo(self):
        sq = self.sq
        dim = pygame.Surface((8 * sq, 8 * sq), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 120))
        self.screen.blit(dim, (self.bx, self.by))
        card, cells = self.promo_layout()
        pygame.draw.rect(self.screen, (24, 26, 34), card, border_radius=14)
        pygame.draw.rect(self.screen, (80, 86, 108), card, 2, border_radius=14)
        self.blit_text("Promote to", self.f_bold, C_TEXT, card.centerx, card.y + int(sq * 0.28), "center")
        mouse = pygame.mouse.get_pos()
        col = self.pos.side
        for t, r in cells:
            if r.collidepoint(mouse):
                pygame.draw.rect(self.screen, (70, 78, 104), r, border_radius=10)
            self.screen.blit(self.piece_surf(t | (col << 3), int(sq * 0.95)), (r.x + int(sq * 0.025), r.y + int(sq * 0.025)))

    def draw_result(self):
        sq = self.sq
        dim = pygame.Surface((8 * sq, 8 * sq), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 140))
        self.screen.blit(dim, (self.bx, self.by))
        card, r1, r2 = self.overlay_buttons()
        pygame.draw.rect(self.screen, (26, 28, 37), card, border_radius=16)
        pygame.draw.rect(self.screen, (86, 92, 116), card, 2, border_radius=16)
        title, sub, _ = self.result
        self.blit_text(title, self.f_big, C_TEXT, card.centerx, card.y + card.h * 0.26, "center")
        self.blit_text(sub, self.f_mid, (150, 205, 110) if "wins" in sub else C_MUTED, card.centerx, card.y + card.h * 0.50, "center")
        mouse = pygame.mouse.get_pos()
        for r, label, acc in ((r1, "New Game", True), (r2, "View Board", False)):
            hov = r.collidepoint(mouse)
            col = (C_ACC_H if hov else C_ACC) if acc else (C_BTN_H if hov else C_BTN)
            pygame.draw.rect(self.screen, col, r, border_radius=10)
            self.blit_text(label, self.f_bold, (255, 255, 255), r.centerx, r.centery, "center")

    # ----------------------------------------------------------------- panel
    def material(self):
        vals = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9}
        start = {1: 8, 2: 2, 3: 2, 4: 2, 5: 1}
        cnt = [[0] * 7 for _ in range(2)]
        for s in SQ64:
            p = self.pos.board[s]
            if p and (p & 7) != 6:
                cnt[p >> 3][p & 7] += 1
        missing = [[max(0, start[t] - cnt[c][t]) for t in range(1, 6)] for c in (0, 1)]
        tot = [sum(cnt[c][t] * vals[t] for t in range(1, 6)) for c in (0, 1)]
        return missing, tot[0] - tot[1]

    def player_name(self, color):
        h = self.human_color()
        if h is None:
            return "White" if color == 0 else "Black"
        return "You" if color == h else "Computer"

    def draw_player_card(self, rect, color, missing, diff):
        S = self.screen
        active = (self.pos.side == color) and not self.result
        pygame.draw.rect(S, C_CARD, rect, border_radius=12)
        if active:
            pygame.draw.rect(S, (104, 160, 72), rect, 2, border_radius=12)
        # avatar
        av = pygame.Rect(rect.x + 10, rect.y + 9, 40, 40)
        pygame.draw.rect(S, (66, 70, 88) if color == 0 else (50, 53, 68), av, border_radius=10)
        S.blit(self.piece_surf(6 | (color << 3), 36), (av.x + 2, av.y + 2))
        name = self.player_name(color)
        self.blit_text(name, self.f_bold, C_TEXT, rect.x + 60, rect.y + 9)
        if name == "Computer":
            self.blit_text(LEVELS[self.level][0], self.f_small, C_MUTED, rect.x + 60 + self.txt(name, self.f_bold, C_TEXT).get_width() + 8, rect.y + 11)
        # captured pieces: those of the opposite colour that are missing
        x = rect.x + 60
        y = rect.y + 31
        for t in (5, 4, 3, 2, 1):
            for _ in range(missing[1 - color][t - 1]):
                S.blit(self.piece_surf(t | ((1 - color) << 3), 22), (x, y))
                x += 11 if t == 1 else 16
        if diff > 0 if color == 0 else diff < 0:
            self.blit_text("+%d" % abs(diff), self.f_small, C_MUTED, x + 8, y + 3)

    def draw_status(self):
        S, r = self.screen, self.r_status
        pygame.draw.rect(S, C_CARD, r, border_radius=10)
        dot = (245, 245, 238) if self.pos.side == 0 else (18, 18, 22)
        if self.result:
            text = self.result[0] + " - " + self.result[1] if self.result[0] == "Checkmate" else self.result[0] + ": " + self.result[1]
            dot = (150, 205, 110)
        elif self.promo:
            text = "Choose a promotion piece"
        elif self.ai_busy:
            dots = "." * (1 + int(time.time() * 3) % 3)
            text = ("Computer is thinking" if self.busy_kind == "move" else "Finding a hint") + dots
        else:
            who = "Your move" if (self.human_color() == self.pos.side) else ("White to move" if self.pos.side == 0 else "Black to move")
            text = who + ("  -  Check!" if self.in_check else "")
        pygame.draw.circle(S, dot, (r.x + 20, r.centery), 8)
        pygame.draw.circle(S, (110, 116, 138), (r.x + 20, r.centery), 8, 1)
        self.blit_text(text, self.f_body, (255, 150, 140) if self.in_check and not self.result else C_TEXT, r.x + 38, r.centery, "midleft")
        if self.ai_busy:
            cx, cy = r.right - 22, r.centery
            for i in range(8):
                a = time.time() * 6 - i * 0.7
                px, py = cx + math.cos(a) * 9, cy + math.sin(a) * 9
                v = int(80 + 175 * (1 - i / 8))
                pygame.draw.circle(S, (v, v, v), (int(px), int(py)), 2)

    def draw_moves(self):
        S, r = self.screen, self.r_moves
        pygame.draw.rect(S, C_CARD, r, border_radius=12)
        inner = r.inflate(-12, -12)
        if not self.sans:
            self.blit_text("Moves will appear here", self.f_body, C_MUTED, r.centerx, r.centery, "center")
            return
        rh = 28
        rows = (len(self.sans) + 1) // 2
        total = rows * rh
        maxs = max(0, total - inner.h)
        if self.autoscroll:
            self.scroll = maxs
            self.autoscroll = False
        self.scroll = max(0, min(self.scroll, maxs))
        S.set_clip(inner)
        col2 = inner.x + 54 + (inner.w - 60) // 2
        for i in range(rows):
            y = inner.y + i * rh - self.scroll
            if y + rh < inner.y or y > inner.bottom:
                continue
            if i % 2 == 0:
                pygame.draw.rect(S, C_CARD2, (inner.x, y, inner.w, rh), border_radius=6)
            self.blit_text("%d." % (i + 1), self.f_small, C_MUTED, inner.x + 10, y + rh // 2, "midleft")
            for j, cx in ((2 * i, inner.x + 54), (2 * i + 1, col2)):
                if j < len(self.sans):
                    if j == len(self.sans) - 1:
                        w = self.txt(self.sans[j], self.f_bold, (255, 255, 255)).get_width() + 14
                        pygame.draw.rect(S, (92, 140, 70), (cx - 7, y + 3, w, rh - 6), border_radius=6)
                        self.blit_text(self.sans[j], self.f_bold, (255, 255, 255), cx, y + rh // 2, "midleft")
                    else:
                        self.blit_text(self.sans[j], self.f_bold, C_TEXT, cx, y + rh // 2, "midleft")
        S.set_clip(None)
        if maxs > 0:
            th = max(24, int(inner.h * inner.h / total))
            ty = inner.y + int((inner.h - th) * self.scroll / maxs)
            pygame.draw.rect(S, (90, 96, 118), (inner.right - 4, ty, 4, th), border_radius=2)

    def draw_button(self, b):
        S, r = self.screen, b["rect"]
        mouse = pygame.mouse.get_pos()
        en = b["enabled"]()
        hov = en and r.collidepoint(mouse)
        if b["accent"]:
            col = C_ACC_H if hov else C_ACC
        else:
            col = C_BTN_H if hov else C_BTN
        if not en:
            col = (44, 47, 60)
        rr = r.move(0, 1) if (self.pressed is b) else r
        pygame.draw.rect(S, col, rr, border_radius=9)
        self.blit_text(b["label"](), self.f_body, (255, 255, 255) if en else (110, 115, 132), rr.centerx, rr.centery, "center")

    def draw_panel(self):
        missing, diff = self.material()
        top_color = 0 if self.flipped else 1
        bot_color = 1 - top_color
        self.draw_player_card(self.r_top, top_color, missing, diff)
        self.draw_status()
        self.draw_moves()
        self.draw_player_card(self.r_bottom, bot_color, missing, diff)
        for b in self.buttons:
            self.draw_button(b)

    def draw(self):
        self.screen.blit(self.bg, (0, 0))
        self.draw_eval_bar()
        self.draw_board()
        if self.promo:
            self.draw_promo()
        elif self.result and self.show_result and not self.anim:
            self.draw_result()
        self.draw_panel()
        if self.toast:
            surf = self.txt(self.toast[0], self.f_bold, (255, 255, 255))
            rr = surf.get_rect()
            chip = pygame.Rect(0, 0, rr.w + 36, rr.h + 18)
            chip.midbottom = (self.bx + 4 * self.sq, self.by + 8 * self.sq - 14)
            layer = pygame.Surface(chip.size, pygame.SRCALPHA)
            pygame.draw.rect(layer, (18, 20, 28, 225), layer.get_rect(), border_radius=chip.h // 2)
            self.screen.blit(layer, chip)
            self.screen.blit(surf, surf.get_rect(center=chip.center))

    # ------------------------------------------------------------------ loop
    def run(self):
        while True:
            self.clock.tick(60)
            sz = self.screen.get_size()
            if sz != (self.W, self.H):
                w, h = max(sz[0], self.MIN_W), max(sz[1], self.MIN_H)
                if (w, h) != sz:
                    self.screen = pygame.display.set_mode((w, h), pygame.RESIZABLE)
                self.layout()
            for ev in pygame.event.get():
                if not self.handle(ev):
                    self.cancel_search()
                    pygame.quit()
                    return
            self.update()
            self.update_cursor()
            self.draw()
            pygame.display.flip()


def main():
    args = sys.argv[1:]
    if "--uci" in args:
        uci_main()
        return
    if "--perft" in args:
        d = int(args[args.index("--perft") + 1]) if len(args) > args.index("--perft") + 1 else 3
        pos = Position()
        t = time.time()
        n = perft(pos, d)
        print("perft(%d) = %d  (%.2fs)" % (d, n, time.time() - t))
        return
    if pygame is None:
        print("Pygame is required for the GUI.  Install it with:\n\n    pip install pygame\n")
        sys.exit(1)
    App().run()


if __name__ == "__main__":
    main()