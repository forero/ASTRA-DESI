from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from .run_fiber_assignment_voids import run_mock_cli
except ImportError:
    from run_fiber_assignment_voids import run_mock_cli

DEFAULT_OUTPUT_DIR = '/pscratch/sd/v/vtorresg/void_catalog/complete_targets'


def main(argv=None):
    return run_mock_cli(argv=argv,
                        default_output=DEFAULT_OUTPUT_DIR,
                        fixed_kind='complete')


if __name__ == '__main__':
    raise SystemExit(main())
