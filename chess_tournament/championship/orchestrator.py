"""
Main championship orchestrator coordinating all tournament stages.
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Any
import logging

from .config import ChampionshipConfig, setup_logging
from .validator import SubmissionValidator
from .runner import TournamentRunner
from .leaderboard import LeaderboardGenerator


class ChessChampionship:
    """
    Main orchestrator for complete chess championship.
    
    Coordinates all stages: Validation → Qualifiers → Semifinals → Finals
    
    Example:
        config = ChampionshipConfig()
        championship = ChessChampionship(config, baseline_factories)
        results = championship.run(submissions_df)
    """
    
    def __init__(self,
                 config: ChampionshipConfig,
                 baseline_factories: Dict[str, Dict[str, Any]],
                 logger: logging.Logger = None):
        """
        Args:
            config: ChampionshipConfig instance
            baseline_factories: Dict mapping baseline_key -> {name, factory}
                              where factory is callable returning a Player
            logger: Logger instance (auto-created if not provided)
        """
        self.config = config
        self.baseline_factories = baseline_factories
        self.logger = logger or setup_logging(config)
    
    def run(self,
            submissions_df: pd.DataFrame,
            qualifiers_group_size: int = 8,
            qualifiers_rounds: int = 3,
            qualifiers_top_k: int = 2,
            semifinals_group_size: int = 8,
            semifinals_rounds: int = 3,
            semifinals_top_k: int = 2,
            finals_games_per_pair: int = 2,
            max_half_moves: int = 200) -> Dict[str, Any]:
        """
        Run complete championship: Qualifiers → Semifinals → Finals
        
        Args:
            submissions_df: DataFrame from Google Forms with columns:
                          - student_number (required)
                          - repo_url (required)
            qualifiers_group_size: Players per group in qualifiers
            qualifiers_rounds: Number of Swiss rounds in qualifiers
            qualifiers_top_k: Top K per group advancing from qualifiers
            semifinals_group_size: Players per group in semifinals
            semifinals_rounds: Number of rounds in semifinals
            semifinals_top_k: Top K per group advancing to finals
            finals_games_per_pair: Games per pair in finals
            max_half_moves: Maximum half-moves per game
        
        Returns:
            Dict with keys:
                - leaderboard: Final results DataFrame
                - leaderboard_csv: Path to CSV
                - leaderboard_md: Path to Markdown
                - work_dir: Output directory path
                - validation_results: Validation DataFrame
        """
        
        self.logger.info("╔════════════════════════════════════════╗")
        self.logger.info("║   CHESS CHAMPIONSHIP FRAMEWORK          ║")
        self.logger.info("║   Qualifiers → Semifinals → Finals      ║")
        self.logger.info("╚════════════════════════════════════════╝\n")
        self.logger.info(f"Work directory: {self.config.work_dir}\n")
        
        # [1] Validate submissions
        self.logger.info("[1/6] VALIDATING SUBMISSIONS")
        validator = SubmissionValidator(self.config, self.logger)
        validation_df = validator.process_submissions(submissions_df)
        
        # [2] Build participants list
        self.logger.info("\n[2/6] BUILDING PARTICIPANTS")
        participants = self._build_participants(validation_df)
        
        # Save participants snapshot for reference
        with open(self.config.participants_snapshot, "w") as f:
            json.dump(participants, f, indent=2, default=str)
        
        num_students = sum(1 for p in participants if p['type'] == 'student')
        num_baselines = sum(1 for p in participants if p['type'] == 'baseline')
        self.logger.info(f"Total participants: {len(participants)} ({num_students} students, {num_baselines} baselines)")
        
        # [3] Qualifiers
        self.logger.info("\n[3/6] QUALIFIERS (Swiss Tournament)")
        runner = TournamentRunner(self.config, self.logger, self.baseline_factories)
        runner.create_plan(participants, qualifiers_group_size, self.config.qualifiers_plan_csv)
        
        qual_results = runner.run_swiss_stage(
            "Qualifiers",
            self.config.qualifiers_plan_csv,
            self.config.qualifiers_results_csv,
            n_rounds=qualifiers_rounds,
            games_per_pairing=1,
            max_half_moves=max_half_moves
        )
        
        advancing = runner.get_advancing(qual_results, qualifiers_top_k)
        self.logger.info(f"Advancing to semifinals: {len(advancing)} players")
        
        # [4] Semifinals
        self.logger.info("\n[4/6] SEMIFINALS (Swiss Tournament)")
        runner.create_plan(advancing, semifinals_group_size, self.config.semifinals_plan_csv)
        
        semi_results = runner.run_swiss_stage(
            "Semifinals",
            self.config.semifinals_plan_csv,
            self.config.semifinals_results_csv,
            n_rounds=semifinals_rounds,
            games_per_pairing=1,
            max_half_moves=max_half_moves
        )
        
        finalists = runner.get_advancing(semi_results, semifinals_top_k)
        self.logger.info(f"Finalists: {len(finalists)} players")
        
        # [5] Finals
        self.logger.info("\n[5/6] FINALS")
        num_finalists = len(finalists) if len(finalists) > 0 else 1
        runner.create_plan(finalists, num_finalists, self.config.finals_plan_csv)
        
        # For finals, use round-robin format
        final_results = runner.run_swiss_stage(
            "Finals",
            self.config.finals_plan_csv,
            self.config.finals_results_csv,
            n_rounds=max(1, num_finalists - 1),
            games_per_pairing=finals_games_per_pair,
            max_half_moves=max_half_moves
        )
        
        # [6] Generate leaderboard
        self.logger.info("\n[6/6] GENERATING LEADERBOARD")
        LeaderboardGenerator.write_markdown(final_results, self.config.final_leaderboard_md)
        final_results.to_csv(self.config.final_leaderboard_csv, index=False)
        
        self.logger.info(f"\n✅ CHAMPIONSHIP COMPLETE!\n")
        self.logger.info(f"Leaderboard (Markdown): {self.config.final_leaderboard_md}")
        self.logger.info(f"Leaderboard (CSV): {self.config.final_leaderboard_csv}")
        self.logger.info(f"\n{'='*80}")
        self.logger.info("FINAL LEADERBOARD".center(80))
        self.logger.info(f"{'='*80}\n")
        self.logger.info(final_results.to_string(index=False))
        self.logger.info(f"\n{'='*80}\n")
        
        return {
            "leaderboard": final_results,
            "leaderboard_csv": self.config.final_leaderboard_csv,
            "leaderboard_md": self.config.final_leaderboard_md,
            "work_dir": self.config.work_dir,
            "validation_results": validation_df
        }
    
    def _build_participants(self, validation_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Build participant descriptors from validation results and baselines.
        
        Args:
            validation_df: Output from SubmissionValidator.process_submissions
        
        Returns:
            List of participant descriptors (student + baseline)
        """
        participants = []
        
        # Add approved students
        approved = validation_df[validation_df["approved"] == True]
        for _, row in approved.iterrows():
            student_num = str(row["student_number"]).strip()
            participants.append({
                "type": "student",
                "id": student_num,
                "name": f"Student-{student_num}",
                "repo_path": row["repo_path"]
            })
        
        # Add baseline players - IMPORTANT: include the factory!
        for baseline_key, info in self.baseline_factories.items():
            participants.append({
                "type": "baseline",
                "id": f"baseline-{baseline_key}",
                "name": info["name"],
                "baseline_key": baseline_key,
                "factory": info["factory"]  # <-- THIS IS THE FIX!
            })
        
        return participants
