"""Comprehensive tests for all pipeline implementations."""

import pytest
import time
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass

from src.orchestration.pipeline import RLPipeline, PipelineConfig, ProblemResult
from src.orchestration.baseline import BaselineRunner, BaselineConfig, BaselineResult
from src.orchestration.simplified_pipeline import SimplifiedRLPipeline, SimplifiedConfig
from src.orchestration.improved_pipeline import ImprovedRLPipeline, ImprovedPipelineConfig


class TestRLPipeline:
    """Tests for RLPipeline (main RL-guided pipeline)."""
    
    def test_pipeline_config_defaults(self):
        """Test default configuration."""
        config = PipelineConfig()
        
        assert config.max_iterations == 50
        assert config.early_stop_score == 0.9
        assert config.mcts.exploration_constant == 1.414
    
    def test_pipeline_config_custom(self):
        """Test custom configuration."""
        from src.rl_controller.mcts import MCTSConfig
        from src.rl_controller.actions import ActionConfig
        
        config = PipelineConfig(
            max_iterations=100,
            early_stop_score=0.85,
            mcts=MCTSConfig(exploration_constant=2.0),
        )
        
        assert config.max_iterations == 100
        assert config.early_stop_score == 0.85
        assert config.mcts.exploration_constant == 2.0
    
    @patch('src.orchestration.pipeline.NVIDIANIMClient')
    @patch('src.orchestration.pipeline.PRMEvaluator')
    @patch('src.orchestration.pipeline.ActionExecutor')
    @patch('src.orchestration.pipeline.MCTSController')
    def test_pipeline_initialization(self, mock_mcts, mock_executor, mock_prm, mock_client, tmp_path):
        """Test pipeline initialization."""
        pipeline = RLPipeline(
            api_key="test_key",
            config=PipelineConfig(),
            results_dir=str(tmp_path),
        )
        
        assert pipeline.config is not None
        assert pipeline.generator is not None
        assert pipeline.evaluator is not None
        assert pipeline.action_executor is not None
        assert pipeline.mcts is not None
    
    def test_problem_result_creation(self):
        """Test ProblemResult dataclass."""
        result = ProblemResult(
            problem_id="test_001",
            problem="Test problem",
            final_answer="Test answer",
            final_score=0.85,
            total_tokens=100,
            total_tokens_input=40,
            total_tokens_output=60,
            total_api_calls=5,
            num_expansions=3,
            num_reflections=2,
            num_backtracks=1,
            max_depth_reached=5,
            latency_seconds=2.5,
            correct=True,
            ground_truth="Test answer",
        )
        
        assert result.problem_id == "test_001"
        assert result.final_score == 0.85
        assert result.total_tokens == 100
        assert result.num_expansions == 3
        assert result.correct is True
    
    def test_compute_aggregate_stats(self, tmp_path):
        """Test aggregate statistics computation."""
        pipeline = RLPipeline.__new__(RLPipeline)
        pipeline._results = []
        
        for i in range(3):
            pipeline._results.append(ProblemResult(
                problem_id=f"p{i}",
                problem=f"Problem {i}",
                final_answer=f"Answer {i}",
                final_score=0.5 + i * 0.1,
                total_tokens_input=40 + i * 10,  # 40, 50, 60
                total_tokens_output=60 + i * 10,  # 60, 70, 80
                total_api_calls=5 + i,
                num_expansions=3 + i,
                num_reflections=2 + i,
                num_backtracks=1 + i,
                latency_seconds=1.0 + i * 0.5,
                correct=i > 0,  # 2 correct, 1 incorrect
            ))
        
        stats = pipeline._compute_aggregate_stats()
        
        assert stats["total_problems"] == 3
        # input: 40+50+60 = 150, output: 60+70+80 = 210, total = 360
        assert stats["total_tokens"] == 360
        assert stats["total_correct"] == 2
        assert stats["accuracy"] == pytest.approx(2/3)


class TestBaselineRunner:
    """Tests for BaselineRunner (zero-shot baseline)."""
    
    def test_baseline_config_defaults(self):
        """Test default baseline configuration."""
        config = BaselineConfig()
        
        assert config.model == "meta/llama-3.1-8b-instruct"
        assert config.temperature == 0.7
        assert config.max_tokens == 2048
    
    def test_baseline_result_creation(self):
        """Test BaselineResult dataclass."""
        result = BaselineResult(
            problem_id="test_001",
            problem="Test problem",
            final_answer="Test answer",
            full_response="Full response text",
            input_tokens=50,
            output_tokens=100,
            total_tokens=150,
            latency_seconds=1.5,
            correct=True,
        )
        
        assert result.problem_id == "test_001"
        assert result.full_response == "Full response text"
        assert result.total_tokens == 150
    
    def test_extract_answer_markers(self):
        """Test answer extraction from various formats."""
        from src.orchestration.baseline import BaselineRunner
        
        runner = BaselineRunner.__new__(BaselineRunner)
        
        # Test "therefore"
        assert "yes" in runner._extract_answer("Therefore, the answer is yes.").lower()
        
        # Test "the answer is"
        assert "42" in runner._extract_answer("The answer is 42.")
        
        # Test markdown format
        assert "paris" in runner._extract_answer("## Answer\n\nParis").lower()


class TestSimplifiedPipeline:
    """Tests for SimplifiedRLPipeline."""
    
    def test_simplified_config_defaults(self):
        """Test default simplified configuration."""
        from src.orchestration.simplified_pipeline import SimplifiedConfig
        
        config = SimplifiedConfig()
        
        assert config.max_steps == 5
        assert config.min_steps == 3
        assert config.backtrack_probability == 0.3
    
    def test_simplified_result_creation(self):
        """Test SimplifiedResult dataclass."""
        from src.orchestration.simplified_pipeline import SimplifiedResult
        
        result = SimplifiedResult(
            problem_id="test_001",
            problem="Test problem",
            final_answer="Test answer",
            final_score=0.85,
            total_tokens=100,
            latency_seconds=2.0,
            backtracks=1,
            paths_explored=2,
            correct=True,
        )
        
        assert result.problem_id == "test_001"
        assert result.backtracks == 1
        assert result.paths_explored == 2


class TestImprovedPipeline:
    """Tests for ImprovedRLPipeline."""
    
    def test_improved_config_defaults(self):
        """Test default improved configuration."""
        config = ImprovedPipelineConfig()
        
        assert config.min_steps_before_conclude == 3
        assert config.explore_even_when_good is True
        assert config.base_backtrack_prob == 0.25
    
    def test_improved_result_creation(self):
        """Test ImprovedProblemResult dataclass."""
        from src.orchestration.improved_pipeline import ImprovedProblemResult
        
        result = ImprovedProblemResult(
            problem_id="test_001",
            problem="Test problem",
            final_answer="Test answer",
            final_score=0.85,
            total_tokens_input=40,
            total_tokens_output=60,
            total_api_calls=5,
            num_expansions=3,
            num_reflections=2,
            num_backtracks=1,
            paths_explored=2,
            latency_seconds=2.5,
            verification_score=0.8,
            verification_confidence=0.9,
            learning_applied=True,
            correct=True,
        )
        
        assert result.problem_id == "test_001"
        assert result.verification_score == 0.8
        assert result.learning_applied is True


class TestRobustPipeline:
    """Tests for RobustRLPipeline."""
    
    def test_robust_config_defaults(self):
        """Test default robust configuration."""
        from src.orchestration.robust_pipeline import RobustPipelineConfig
        
        config = RobustPipelineConfig()
        
        assert config.min_steps_before_conclude == 3
        assert config.max_steps == 8
        assert config.use_beam_search is True
        assert config.beam_width == 3
    
    def test_reasoning_state_creation(self):
        """Test ReasoningState dataclass."""
        from src.orchestration.robust_pipeline import ReasoningState
        
        state = ReasoningState(
            problem_hash="abc123",
            steps=["Step 1", "Step 2"],
            scores=[0.8, 0.9],
            is_complete=False,
            final_answer="",
            final_score=0.0,
        )
        
        assert state.problem_hash == "abc123"
        assert len(state.steps) == 2
        assert state.is_complete is False


class TestSelfReflectionPipeline:
    """Tests for SelfReflectionPipeline."""
    
    def test_self_reflection_config_defaults(self):
        """Test default self-reflection configuration."""
        from src.orchestration.self_reflection_pipeline import SelfReflectionConfig
        
        config = SelfReflectionConfig()
        
        assert config.min_reasoning_steps == 2
        assert config.max_reasoning_steps == 5
        assert config.reflection_depth == 2
        assert config.enable_selective_reflection is True
    
    def test_reflection_step_creation(self):
        """Test ReflectionStep dataclass."""
        from src.orchestration.self_reflection_pipeline import ReflectionStep
        
        step = ReflectionStep(
            step_type="reflection",
            content="This step has a logical flaw",
            critique="The reasoning assumes X without evidence",
            is_valid=False,
            issues_found=["Assumption without evidence"],
        )
        
        assert step.step_type == "reflection"
        assert step.is_valid is False
        assert len(step.issues_found) == 1


class TestAdaptiveReflectionPipeline:
    """Tests for AdaptiveReflectionPipeline."""
    
    def test_adaptive_config_defaults(self):
        """Test default adaptive configuration."""
        from src.orchestration.adaptive_reflection_pipeline import AdaptiveReflectionConfig
        
        config = AdaptiveReflectionConfig()
        
        assert config.low_complexity_threshold == 0.3
        assert config.high_complexity_threshold == 0.7
        assert config.min_reflections == 1
        assert config.max_reflections == 5
        assert config.enable_cross_validation is True
    
    def test_complexity_score_creation(self):
        """Test ComplexityScore dataclass."""
        from src.orchestration.adaptive_reflection_pipeline import ComplexityScore
        
        score = ComplexityScore(
            overall_score=0.65,
            factors={"reasoning_depth": 0.7, "domain_knowledge": 0.6},
            recommended_depth=3,
            reasoning="Multi-step reasoning with domain knowledge",
        )
        
        assert score.overall_score == 0.65
        assert score.recommended_depth == 3
        assert "reasoning_depth" in score.factors
    
    def test_reflection_checkpoint_creation(self):
        """Test ReflectionCheckpoint dataclass."""
        from src.orchestration.adaptive_reflection_pipeline import ReflectionCheckpoint
        
        checkpoint = ReflectionCheckpoint(
            step=2,
            reasoning_chain=["Step 1", "Step 2"],
            answer="Yes",
            confidence=0.85,
            issues_found=["Issue 1"],
            timestamp=time.time(),
        )
        
        assert checkpoint.step == 2
        assert checkpoint.confidence == 0.85
        assert len(checkpoint.reasoning_chain) == 2


class TestDPOComponents:
    """Tests for DPO trainer components."""
    
    def test_preference_pair_creation(self):
        """Test PreferencePair dataclass."""
        from src.rl_controller.dpo_trainer import PreferencePair
        
        pair = PreferencePair(
            problem="Test problem",
            chosen_path=["Step 1", "Step 2"],
            rejected_path=["Wrong step"],
            chosen_score=0.9,
            rejected_score=0.3,
            metadata={"source": "test"},
        )
        
        assert pair.problem == "Test problem"
        assert len(pair.chosen_path) == 2
        assert pair.chosen_score > pair.rejected_score
    
    def test_dpo_config_defaults(self):
        """Test DPO configuration."""
        from src.rl_controller.dpo_trainer import DPOConfig
        
        config = DPOConfig()
        
        assert config.beta == 0.1
        assert config.learning_rate == 5e-6
        assert config.batch_size == 4
        assert config.n_epochs == 3
    
    def test_preference_dataset_operations(self):
        """Test PreferenceDataset operations."""
        from src.rl_controller.dpo_trainer import PreferenceDataset
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = PreferenceDataset(storage_path=f"{tmpdir}/test_pairs.json")
            
            # Add pairs
            dataset.add_pair(
                problem="Test",
                chosen_path=["Good path"],
                rejected_path=["Bad path"],
                chosen_score=0.9,
                rejected_score=0.3,
            )
            
            assert len(dataset.pairs) == 1
            
            # Get batch
            batch = dataset.get_batch(batch_size=1)
            assert len(batch) == 1


class TestPolicyLearning:
    """Tests for policy learning components."""
    
    def test_policy_network_creation(self):
        """Test PolicyNetwork instantiation."""
        from src.rl_controller.policy_learning import PolicyNetwork
        
        network = PolicyNetwork(
            state_dim=128,
            hidden_dim=256,
            n_actions=4,
        )
        
        assert network is not None
    
    def test_policy_learner_creation(self):
        """Test PolicyLearner instantiation."""
        from src.rl_controller.policy_learning import PolicyLearner
        
        learner = PolicyLearner(state_dim=128)
        
        assert learner is not None
        assert learner.gamma == 0.95


class TestAsyncBatchPipeline:
    """Tests for async batch processing."""
    
    def test_batch_config_defaults(self):
        """Test batch configuration."""
        from src.orchestration.async_batch_pipeline import BatchConfig
        
        config = BatchConfig()
        
        assert config.max_concurrent == 10
        assert config.checkpoint_interval == 10
        assert config.error_handling == "continue"
    
    def test_batch_result_creation(self):
        """Test BatchResult dataclass."""
        from src.orchestration.async_batch_pipeline import BatchResult
        from datetime import datetime
        
        result = BatchResult(
            total_problems=10,
            successful=8,
            failed=2,
            results=[],
            errors=[{"problem_id": "p1", "error": "timeout"}],
            total_tokens=5000,
            total_latency_seconds=30.0,
            avg_latency_per_problem=3.0,
        )
        
        assert result.total_problems == 10
        assert result.successful == 8
        assert result.avg_latency_per_problem == 3.0
