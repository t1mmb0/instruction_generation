from typing import Dict, List

from src.utils.general_utils import set_seed
from src.runners.single_run import SingleRunExecutor
from src.runners.run_config import RunConfig

class ExperimentRunner:
    def __init__(
        self,
        executor: SingleRunExecutor,
        config: RunConfig,
    ):
        self.executor = executor
        self.config = config
        self.results: Dict[int, dict] = {}

    def run(self) -> Dict[int, dict]:
        self.results.clear()

        total_runs = len(self.config.seeds)
        for run_idx, seed in enumerate(self.config.seeds, start = 1):
            print("=" * 60)
            print(f"RUN {run_idx} / {total_runs}   (seed = {seed})")
            print("=" * 60)     
            set_seed(seed)
            run_result = self.executor.run(seed=seed,
                                           config=self.config,
            )
            self.results[seed] = run_result

        return self.results

        
        
