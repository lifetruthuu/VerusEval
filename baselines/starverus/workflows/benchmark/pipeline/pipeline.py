from __future__ import annotations

import logging
import json
import shutil
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from .agents import (
    Actor,
    Aligner,
    BestCandidateSelector,
    Judge,
    Planner,
    Repairer,
    Rewriter,
    Verifier,
    format_errors,
)
from .history import RepairHistory
from .schemas import PipelineResult, RepairHistoryEntry, VerificationResult


CheckpointWriter = Callable[[str, str], Path]


def _setup_logger(save_dir: Path) -> logging.Logger:
    name = f"starverus_{abs(hash(str(save_dir.resolve())))}"
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(save_dir / "repair.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    return logger


class ContractAlignment:
    def __init__(
        self,
        judge: Judge,
        aligner: Aligner,
        verifier: Verifier,
        max_iterations: int,
        logger: logging.Logger,
    ):
        self.judge = judge
        self.aligner = aligner
        self.verifier = verifier
        self.max_iterations = max_iterations
        self.logger = logger

    def run(
        self,
        current: VerificationResult,
        save_checkpoint: CheckpointWriter,
    ) -> Tuple[VerificationResult, int]:
        self.logger.info("===== Contract Alignment: Judge -> Aligner =====")
        iterations = 0
        for iteration in range(1, self.max_iterations + 1):
            iterations = iteration
            code = current.path.read_text(encoding="utf-8")
            aligned, diagnosis = self.judge.evaluate(code)
            if aligned:
                self.logger.info("Judge: contract aligned at iteration %d", iteration)
                break

            self.logger.info("Judge: contract misaligned at iteration %d", iteration)
            aligned_code = self.aligner.align(code, diagnosis)
            if not aligned_code.strip():
                raise RuntimeError("Aligner returned empty code")
            checkpoint = save_checkpoint(
                aligned_code,
                f"contract_alignment_{iteration}",
            )
            current = self.verifier.verify(checkpoint)
            if current.is_verified:
                self.logger.info("Verifier: code verified during contract alignment")
                break
        return current, iterations


class ProofRepair:
    def __init__(
        self,
        planner: Planner,
        repairer: Repairer,
        actor: Actor,
        rewriter: Rewriter,
        verifier: Verifier,
        selector: BestCandidateSelector,
        history: RepairHistory,
        save_dir: Path,
        max_epochs: int,
        num_actor_candidates: int,
        max_rewriter_invocations: int,
        additional_epochs: int,
        logger: logging.Logger,
    ):
        self.planner = planner
        self.repairer = repairer
        self.actor = actor
        self.rewriter = rewriter
        self.verifier = verifier
        self.selector = selector
        self.history = history
        self.save_dir = save_dir
        self.max_epochs = max_epochs
        self.num_actor_candidates = num_actor_candidates
        self.max_rewriter_invocations = max_rewriter_invocations
        self.additional_epochs = additional_epochs
        self.logger = logger

    @staticmethod
    def _should_accept(
        current: VerificationResult,
        candidate: VerificationResult,
        target_error_type: str,
    ) -> Tuple[bool, str]:
        if current.is_compilable and not candidate.is_compilable:
            return False, "compilation regression"
        if candidate.is_verified:
            return True, "verified"
        if target_error_type not in candidate.errors_by_type:
            return True, f"resolved {target_error_type}"
        if (
            candidate.verified_obligations >= current.verified_obligations
            and candidate.errors <= current.errors
        ):
            return True, "score stable or improved"
        return False, "score regressed"

    def _run_epochs(
        self,
        current: VerificationResult,
        repair_cycle: int,
        epoch_budget: int,
        first_epoch: int,
        save_checkpoint: CheckpointWriter,
    ) -> Tuple[VerificationResult, int]:
        epochs_used = 0
        for local_epoch in range(1, epoch_budget + 1):
            if current.is_verified:
                break
            epoch = first_epoch + epochs_used
            epochs_used += 1
            code = current.path.read_text(encoding="utf-8")

            decision = self.planner.plan(code, current)
            if decision.used_fallback:
                self.logger.warning("Planner fallback: %s", decision.summary)
            self.logger.info(
                "Epoch %d: Planner selected %s",
                epoch,
                decision.selected_repairer,
            )

            blueprint = self.repairer.create_blueprint(
                code,
                current,
                decision,
                self.history.for_repairer(),
            )
            candidate_codes = self.actor.act(
                code,
                blueprint,
                self.num_actor_candidates,
            )
            if not candidate_codes:
                raise RuntimeError("Actor returned no repair candidates")

            epoch_dir = (
                self.save_dir
                / "proof_repair"
                / f"cycle_{repair_cycle}"
                / f"epoch_{local_epoch}"
            )
            epoch_dir.mkdir(parents=True, exist_ok=True)
            (epoch_dir / "planner_decision.json").write_text(
                json.dumps(asdict(decision), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (epoch_dir / "repair_blueprint.txt").write_text(
                blueprint.content,
                encoding="utf-8",
            )
            evaluated = []
            for index, candidate_code in enumerate(candidate_codes):
                candidate_path = epoch_dir / f"actor_candidate_{index}.rs"
                candidate_path.write_text(candidate_code, encoding="utf-8")
                result = self.verifier.verify(candidate_path)
                evaluated.append(result)
                self.logger.info(
                    "Actor candidate %d score: [%d, %d]",
                    index,
                    result.verified_obligations,
                    result.errors,
                )

            best = self.selector.select(evaluated)
            accepted, result_reason = self._should_accept(
                current,
                best,
                decision.root_cause_error_type,
            )
            if accepted:
                checkpoint = save_checkpoint(
                    best.path.read_text(encoding="utf-8"),
                    f"proof_repair_cycle_{repair_cycle}_epoch_{local_epoch}",
                )
                current = replace(best, path=checkpoint)

            feedback_source = current if accepted else best
            self.history.append(
                RepairHistoryEntry(
                    epoch=epoch,
                    repair_cycle=repair_cycle,
                    repairer_name=decision.selected_repairer,
                    error_type=decision.root_cause_error_type,
                    blueprint=blueprint.content,
                    verifier_feedback=format_errors(
                        feedback_source.errors_by_type
                    ),
                    result=result_reason,
                    verified_obligations=feedback_source.verified_obligations,
                    errors=feedback_source.errors,
                    accepted=accepted,
                    planner_summary=decision.summary,
                )
            )
            self.logger.info(
                "Epoch %d result: %s (%s)",
                epoch,
                "accepted" if accepted else "rejected",
                result_reason,
            )
        return current, epochs_used

    def run(
        self,
        current: VerificationResult,
        save_checkpoint: CheckpointWriter,
    ) -> Tuple[VerificationResult, int, int]:
        self.logger.info(
            "===== Proof Repair: Planner -> Repairer -> Actor -> Verifier ====="
        )
        total_epochs = 0
        rewriter_invocations = 0
        current, epochs_used = self._run_epochs(
            current,
            repair_cycle=0,
            epoch_budget=self.max_epochs,
            first_epoch=1,
            save_checkpoint=save_checkpoint,
        )
        total_epochs += epochs_used

        while (
            not current.is_verified
            and rewriter_invocations < self.max_rewriter_invocations
        ):
            rewriter_invocations += 1
            self.logger.info("===== Rewriter invocation %d =====", rewriter_invocations)
            rewritten_code = self.rewriter.rewrite(
                current.path.read_text(encoding="utf-8"),
                current,
            )
            if not rewritten_code.strip():
                raise RuntimeError("Rewriter returned empty code")
            checkpoint = save_checkpoint(
                rewritten_code,
                f"rewriter_{rewriter_invocations}",
            )
            current = self.verifier.verify(checkpoint)
            if current.is_verified:
                break

            current, epochs_used = self._run_epochs(
                current,
                repair_cycle=rewriter_invocations,
                epoch_budget=self.additional_epochs,
                first_epoch=total_epochs + 1,
                save_checkpoint=save_checkpoint,
            )
            total_epochs += epochs_used

        return current, total_epochs, rewriter_invocations


class StarVerusPipeline:
    def __init__(
        self,
        source_file: Path,
        save_dir: Path,
        config: Dict[str, Any],
        backbone_model: str,
        components: Optional[Dict[str, Any]] = None,
    ):
        self.source_file = source_file
        self.save_dir = save_dir
        self.config = config
        self.backbone_model = backbone_model
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.trace_dir = self.save_dir / "repair_trace"
        self.trace_dir.mkdir(exist_ok=True)
        self.logger = _setup_logger(self.save_dir)
        self.checkpoint_index = 0

        role_models = config.get("models", {})

        def model_for(role: str) -> str:
            return role_models.get(role) or backbone_model

        repair_temperature = float(config["proof_repair"]["temperature"])
        provided = components or {}
        self.verifier = provided.get("verifier") or Verifier()
        self.selector = provided.get("selector") or BestCandidateSelector()
        self.judge = provided.get("judge") or Judge(
            model_for("judge"), repair_temperature
        )
        self.aligner = provided.get("aligner") or Aligner(
            model_for("aligner"), repair_temperature
        )
        self.planner = provided.get("planner") or Planner(
            model_for("planner"), repair_temperature
        )
        self.repairer = provided.get("repairer") or Repairer(
            model_for("repairer"), repair_temperature
        )
        self.actor = provided.get("actor") or Actor(
            model_for("actor"), repair_temperature
        )
        self.rewriter = provided.get("rewriter") or Rewriter(
            model_for("rewriter"), repair_temperature
        )

    def _save_checkpoint(self, code: str, tag: str) -> Path:
        path = self.trace_dir / f"checkpoint_{self.checkpoint_index}_{tag}.rs"
        path.write_text(code, encoding="utf-8")
        self.checkpoint_index += 1
        return path

    def run(self) -> PipelineResult:
        source_copy = self.save_dir / "source.rs"
        shutil.copy(self.source_file, source_copy)
        initial_path = self.trace_dir / "error.rs"
        shutil.copy(self.source_file, initial_path)

        initial = self.verifier.verify(initial_path)
        current = initial
        self.logger.info(
            "Initial score: [%d, %d]",
            initial.verified_obligations,
            initial.errors,
        )

        alignment_iterations = 0
        repair_epochs = 0
        rewriter_invocations = 0
        history = RepairHistory(self.save_dir / "repair_history.json")

        if not current.is_verified:
            contract_alignment = ContractAlignment(
                self.judge,
                self.aligner,
                self.verifier,
                int(self.config["contract_alignment"]["max_iterations"]),
                self.logger,
            )
            current, alignment_iterations = contract_alignment.run(
                current,
                self._save_checkpoint,
            )

        if not current.is_verified:
            proof_repair = ProofRepair(
                planner=self.planner,
                repairer=self.repairer,
                actor=self.actor,
                rewriter=self.rewriter,
                verifier=self.verifier,
                selector=self.selector,
                history=history,
                save_dir=self.save_dir,
                max_epochs=int(self.config["proof_repair"]["max_epochs"]),
                num_actor_candidates=int(
                    self.config["actor"]["num_candidates"]
                ),
                max_rewriter_invocations=int(
                    self.config["rewriter"]["max_invocations"]
                ),
                additional_epochs=int(
                    self.config["rewriter"]["additional_epochs"]
                ),
                logger=self.logger,
            )
            current, repair_epochs, rewriter_invocations = proof_repair.run(
                current,
                self._save_checkpoint,
            )

        if current.is_verified:
            shutil.copy(current.path, self.save_dir / "correct.rs")

        self.logger.info("========== StarVerus Summary ==========")
        self.logger.info("Verified: %s", current.is_verified)
        self.logger.info("Contract alignment iterations: %d", alignment_iterations)
        self.logger.info("Proof repair epochs: %d", repair_epochs)
        self.logger.info("Rewriter invocations: %d", rewriter_invocations)
        self.logger.info(
            "Final score: [%d, %d]",
            current.verified_obligations,
            current.errors,
        )

        return PipelineResult(
            verified=current.is_verified,
            final_path=current.path,
            initial_score=(initial.verified_obligations, initial.errors),
            final_score=(current.verified_obligations, current.errors),
            alignment_iterations=alignment_iterations,
            repair_epochs=repair_epochs,
            rewriter_invocations=rewriter_invocations,
            history=history.entries,
        )
