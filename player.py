from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import chess
import re
from peft import PeftModel

from chess_tournament import Player

class TransformerPlayer(Player):
    def __init__(self, name: str, model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        super().__init__(name)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.last_move = None

        print(f"[{self.name}] Loading model {model_id} in 4-bit")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )

        # Load your trained LoRA adapter
        self.model = PeftModel.from_pretrained(
            base_model,
            "../qwen-chess-tactics-final"
        )

        self.model.eval()

        self.piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }
        self.plan_active = True
        self.plan_name = None # 'scholars_white' or 'e5_black'

    def get_opening_move(self, board: chess.Board) -> str:
        """Returns a hardcoded move if the plan is active and valid, else None."""
        self.planb = False
        if not self.plan_active:
            return None

        # Determine color and set plan on move 1
        if board.fullmove_number == 1:
            if board.turn == chess.WHITE:
                self.plan_name = 'scholars_white'
            else:
                self.plan_name = 'e5_black'

        # Scholars mate idea
        if self.plan_name == 'scholars_white':
            # Step 1: 1. e4
            if board.fullmove_number == 1:
                return "e2e4"

            # Step 2: 2. Bc4 (Check if f1c4 is legal and d5/e6 haven't blocked path)
            if board.fullmove_number == 2:
                move = chess.Move.from_uci("f1c4")
                if move in board.legal_moves:
                    return "f1c4"

            # Step 3: 3. Qh5 (Check if g6 or Nf6 has been played)
            if board.fullmove_number == 3:
                move = chess.Move.from_uci("d1h5")
                # Abort if Black played g6 or Nf6 which attacks/blocks h5
                if move in board.legal_moves and not board.is_attacked_by(chess.BLACK, chess.H5):
                    return "d1h5"
                move = chess.Move.from_uci("d1f3")
                if move in board.legal_moves and not board.is_attacked_by(chess.BLACK, chess.F3):
                    self.planb = True
                    return "d1f3"


            # Step 4: 4. Qxf7#
            if board.fullmove_number == 4 and not self.planb:
                move = chess.Move.from_uci("h5f7")
                if move in board.legal_moves:
                    return "h5f7"
                move = chess.Move.from_uci("h5f3")
                if move in board.legal_moves:
                    self.planb = True
                    return "h5f3"
                    

            elif board.fullmove_number == 4 and self.planb:
                move = chess.Move.from_uci("f3f7")
                attackers = board.attackers(chess.WHITE, chess.F7)
                if move in board.legal_moves and attackers == 2:
                    return "f3f7"
            # step 5 if we can via alt route
            if board.fullmove_number == 5 and self.planb:
                move = chess.Move.from_uci("f3f7")
                attackers = board.attackers(chess.WHITE, chess.F7)
                if move in board.legal_moves and attackers == 2:
                    return "f3f7"
            

        # Scholars mate idea as black
        if self.plan_name == 'e5_black':

            # Step 1:  e5
            if board.fullmove_number == 1:
                move = chess.Move.from_uci("e7e5")
                return "e7e5"

            # Step 2:  Bc5
            if board.fullmove_number == 2:
                move = chess.Move.from_uci("f8c5")
                if move in board.legal_moves:
                    return "f8c5"

            # Step 3: Qh4
            if board.fullmove_number == 3:
                move = chess.Move.from_uci("d8h4")

                # Abort if white already defends f2 or attacks h4 
                white_moves = [m.uci() for m in board.move_stack]

                if (
                    move in board.legal_moves
                    and not board.is_attacked_by(chess.WHITE, chess.H4)
                    and "d1e2" not in white_moves
                ):
                    return "d8h4"

            # Step 4: Qxf2#
            if board.fullmove_number == 4:
                move = chess.Move.from_uci("h4f2")
                attackers = board.attackers(chess.BLACK, chess.F2)
                if move in board.legal_moves and attackers == 2:
                   return "h4f2"

            # If we reach here, the specific plan steps are exhausted or blocked
            self.plan_active = False
            return None

    def get_move(self, fen: str) -> str:
        board = chess.Board(fen)

        opening_move = self.get_opening_move(board)
        if opening_move:
            print('Plan A')
            self.last_move = opening_move
            return opening_move

        is_endgame = len(board.piece_map()) <= 12

        # Categorized moves
        mate_moves = []
        check_moves = []
        safe_captures = []
        positional_moves = [] # Developing pieces, pushing pawns
        danger_moves = []     # Moves that hang a piece

        for move in board.legal_moves:
            u = move.uci()

            # 1. Lookahead: What happens if we make this move?
            board.push(move)

            # Immediate win detection
            if board.is_checkmate():
                mate_moves.append(u)

            # Blunder detection: Does this move allow the opponent to mate us or take our Queen for free?
            is_blunder = False
            if not board.is_checkmate(): # If we didn't just win...
                for opp_move in board.legal_moves:
                    if board.is_capture(opp_move):
                        cap = board.piece_at(opp_move.to_square)
                        # If opponent can take our Queen/Rook for free next turn
                        if cap and self.piece_values[cap.piece_type] >= 5:
                            # Simple check: Is the piece undefended?
                            if not board.is_attacked_by(board.turn, opp_move.to_square):
                                is_blunder = True
                    if board.is_checkmate():
                        is_blunder = True

            if is_blunder:
                danger_moves.append(u)
                board.pop()
                continue

            # 2. Check detection
            if board.is_check():
                check_moves.append(u)

            board.pop()

            # 3. Static Capture Analysis
            if board.is_capture(move):
                captured_piece = board.piece_at(move.to_square) or (board.is_en_passant(move) and chess.Piece(chess.PAWN, not board.turn))
                if captured_piece:
                    cap_val = self.piece_values[captured_piece.piece_type]
                    move_val = self.piece_values[board.piece_at(move.from_square).piece_type]

                    # If we take something more valuable, or equal value
                    if cap_val >= move_val:
                        safe_captures.append(u)
                    # If piece is undefended, any capture is good
                    elif not board.is_attacked_by(not board.turn, move.to_square):
                        safe_captures.append(u)

            # 4. Endgame/Positional logic
            if is_endgame:
                # Reward pushing pawns closer to promotion
                piece = board.piece_at(move.from_square)
                if piece.piece_type == chess.PAWN:
                    rank = chess.square_rank(move.to_square)
                    if (board.turn == chess.WHITE and rank > 1) or (board.turn == chess.BLACK and rank < 8):
                        positional_moves.append(u)
                # Reward King movement toward the center
                if piece.piece_type == chess.KING:
                    center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
                    if move.to_square in center_squares:
                        positional_moves.append(u)

        # Filter legal moves to remove blunders if possible
        refined_legal = [m.uci() for m in board.legal_moves if m.uci() not in danger_moves]
        if not refined_legal: # If every move is a "blunder", go back to all legal
            refined_legal = [m.uci() for m in board.legal_moves]

        # Construct prompt
        prompt_parts = [f"Current Chess FEN: {fen}"]
        prompt_parts.append(f"Safe moves: {', '.join(refined_legal)}") 

        if mate_moves: prompt_parts.append(f"CHECKMATE in 1: {', '.join(mate_moves)}")
        if check_moves: prompt_parts.append(f"Check moves: {', '.join(check_moves)}")
        if safe_captures: prompt_parts.append(f"Profitable captures: {', '.join(safe_captures)}")
        if positional_moves: prompt_parts.append(f"Strategic/Endgame moves: {', '.join(positional_moves)}")

        prompt_parts.append("Task: Pick the best move. Priority: Mate > Check > Capture > Positional. Output ONLY the UCI move.")

        full_prompt = "\n".join(prompt_parts)

        messages = [{"role": "user", "content": full_prompt}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=10,
                do_sample=True,
                temperature=0.4,
                top_p=0.9,
                num_return_sequences=5
            )

        responses = self.tokenizer.batch_decode(
            generated_ids[:, model_inputs.input_ids.shape[-1]:],
            skip_special_tokens=True
)
        # Use multi responses and pick the best one
        for response in responses:

            match = re.search(r"\b[a-h][1-8][a-h][1-8][qrbn]?\b", response.lower())

            if match:
                move = match.group(0)

                if move in mate_moves:
                    self.last_move = move
                    return move

                if move in check_moves:
                    self.last_move = move
                    return move

                if move in safe_captures:
                    self.last_move = move
                    return move

                if move in refined_legal:
                    self.last_move = move
                    return move
        return None