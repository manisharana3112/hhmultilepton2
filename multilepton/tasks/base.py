# coding: utf-8

"""
Custom base tasks for HH -> Multileptons.
"""

import luigi

from columnflow.tasks.framework.base import BaseTask


class MultileptonTask(BaseTask):

    task_namespace = "multilepton"

    # add a parameter that can be set on the command line
    limit_dataset_files = luigi.IntParameter(
        default=-1,  # -1 means "no limit"
        description="Limit number of dataset files to process. -1 means no limit.",
    )
