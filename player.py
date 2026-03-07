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

        print(f"[{self.name}] Loading model {model_id} in 4-bit...")

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
            "./qwen-chess-tactics-final"
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

    def get_move(self, fen: str) -> str:
        board = chess.Board(fen)

        legal_moves = []
        good_captures = []
        check_moves = []
        mate_moves = []

        for move in board.legal_moves:
            u = move.uci()

            # Prevent reversing previous move
            if self.last_move:
                if u[:2] == self.last_move[2:4] and u[2:4] == self.last_move[:2]:
                    continue

            legal_moves.append(u)

            # ---- capture evaluation ----
            if board.is_capture(move):

                captured_piece = board.piece_at(move.to_square)

                if captured_piece is None and board.is_en_passant(move):
                    captured_value = self.piece_values[chess.PAWN]

                elif captured_piece:
                    captured_value = self.piece_values[captured_piece.piece_type]

                else:
                    captured_value = 0

                moving_piece = board.piece_at(move.from_square)
                moving_value = self.piece_values[moving_piece.piece_type]

                if captured_value > moving_value:
                    good_captures.append(u)

            # ---- check / mate detection ----
            board.push(move)

            if board.is_checkmate():
                mate_moves.append(u)
            elif board.is_check():
                check_moves.append(u)

            board.pop()

        if not legal_moves:
            legal_moves = [m.uci() for m in board.legal_moves]

        capture_text = ""
        if good_captures:
            capture_text = f"\nGood capture moves: {', '.join(good_captures)}"

        check_text = ""
        if check_moves:
            check_text = f"\nCheck moves: {', '.join(check_moves)}"

        mate_text = ""
        if mate_moves:
            mate_text = f"\nCheckmate moves: {', '.join(mate_moves)}"

        prompt = (
            f"Current Chess FEN: {fen}\n"
            f"Legal moves: {', '.join(legal_moves)}"
            f"{capture_text}"
            f"{check_text}"
            f"{mate_text}\n"
            "Task: Pick the best chess move, ALWAYS pick a mate or check if available. Output ONLY the move in UCI format."
        )

        messages = [{"role": "user", "content": prompt}]

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
                temperature=0.7,
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

                if move in good_captures:
                    self.last_move = move
                    return move

                if move in legal_moves:
                    self.last_move = move
                    return move
        return None