import os

import torch


# Load offline-selected rocBLAS/hipBLASLt algorithms.  Runtime tuning and file
# writes stay disabled; every workload shape has an entry in this table.
_tunable = torch.cuda.tunable
_tunable.write_file_on_exit(False)
_tunable.set_filename(os.path.join(os.path.dirname(__file__), "tunable.csv"))
_tunable.tuning_enable(False)
_tunable.enable(True)
_tunable.enable(False)
_using_tuned_backend = False


def run(A, B):
    global _using_tuned_backend
    want_tuned_backend = A.shape[0] >= 172 or A.shape[0] == 2
    if want_tuned_backend != _using_tuned_backend:
        _tunable.enable(want_tuned_backend)
        _using_tuned_backend = want_tuned_backend
    return torch.mm(A, B.T)
