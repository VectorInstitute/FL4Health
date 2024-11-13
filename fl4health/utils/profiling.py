import yappi
import fnmatch

from pathlib import Path


class Profiler:
    def __init__(
            self,
            save_path: Path = "profile_results.prof",
            out_path: Path = "profile_results.out",
            filter_patterns: list[str] = ["*/FL4Health/fl4health/*", "*/site-packages/fl4health/*"],
            clock_type: str = "wall"
    ) -> None:
        """
        Yappi-based deterministic profiler that stores stats related to specified modules.

        Args:
            save_path (Path): The path to save the profile. Saved in the the pstat format.\
                Defaults to profile_results.prof.
            out_path (Path): The path to save the human readable profiling results.\
                Defaults to profile_results.out.
            filter_patterns (List[str]): List of shell-style wildcards to filter function calls
                in the program trace. Defaults to list of "*/FL4Health/fl4health/*" and
                "*/site-packages/fl4health/*" to account for both local development or leveraging
                fl4health through PyPi installation.
            clock_type (string): One of 'wall' or 'cpu'. 'cpu' considers active computation
                time of the CPU, excluding waiting periods. 'wall' time considers the entire duration
                of an operation, including both active execution and waiting times. Defaults to 'wall'.
        """
        self.save_path = save_path
        self.out_path = out_path
        self.filter_patterns = filter_patterns
        self.clock_type = clock_type

    def start(self) -> None:
        """
        Sets the clock type and starts yappi profiler.
        """

        yappi.set_clock_type(self.clock_type)
        yappi.start()

    def stop(self) -> None:
        """
        Stops the yappi profiler and dumps result to the profiler save file and output file.
        """
        yappi.stop()

        # Filter to modules that match specified patterns
        stats = yappi.get_func_stats(
            filter_callback=lambda stat:
                any(fnmatch.fnmatch(stat.module, pattern) for pattern in self.filter_patterns)
        )
        stats = stats.strip_dirs()

        if not stats:
            modules_string = " or ".join(self.filter_patterns)
            raise ValueError(f"Trace does not include calls to specififed modules: {modules_string}")

        with open(self.out_path, "w") as f:
            stats.print_all(
                f,
                columns={
                    0: ("name", 64),
                    1: ("ncall", 5),
                    2: ("tsub", 8),
                    3: ("ttot", 8),
                    4: ("tavg", 8)
                }
            )

        stats.save(self.save_path, type="pstat")
