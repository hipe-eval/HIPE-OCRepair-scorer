from .ocrepair_eval import Evaluation, mer_from_counts, align_records, norm
from .cli import main as cli_entry

__version__ = "0.9.7"
__all__ = ["Evaluation", "align_records","norm"]
