# Grandmaster Chess

A complete chess game with an AI opponent, in a **single Python file**. Pieces are drawn in code, so there are no image or sound assets.

## Features
- Full rules: castling, en passant, promotion, check, checkmate, stalemate, threefold repetition, fifty-move rule, insufficient material
- Engine: 0x88 board, Zobrist hashing, iterative-deepening alpha-beta (PVS), transposition table, null-move pruning, late-move reductions, killer/history heuristics, quiescence search
- 4 difficulty levels, hints, undo, board flip, 4 board themes
- Anti-aliased procedural pieces, smooth animations, drag and drop or click to move, evaluation bar, move list (SAN), PGN export
- Runs as a UCI engine (`--uci`) for Arena, CuteChess, etc.

## Run
```bash
pip install -r requirements.txt
python chess_game.py
```

Other modes:
```bash
python chess_game.py --uci       # UCI engine
python chess_game.py --perft 4   # move-generator test (expects 197281)
```

## Controls
| Key | Action |
|-----|--------|
| N | New game |
| U / Ctrl+Z | Undo |
| F | Flip board |
| H | Hint |
| T | Change theme |
| M | Toggle sound |
| S | Save PGN |
| Esc | Cancel selection |

## License
MIT
