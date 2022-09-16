# -*- coding: utf-8 -*-
import pathlib

import click
from mpi4py import MPI

from .. import log
from ..analyze import run_report
from ..log import print_banner, setup_console
from ..postfit import Postfit
from ..runner import Runner
from .base import base_command, root_path

runcard_path = click.option(
    "-p",
    "--runcard_path",
    type=click.Path(path_type=pathlib.Path),
    default=root_path / "runcards",
    required=False,
    help="path to runcard",
)


fit_card = click.option(
    "-f", "--fit_card", type=str, default=None, required=True, help="fit card name",
)

n_replica = click.option(
    "-n",
    "--n_replica",
    type=int,
    default=None,
    required=True,
    help="Number of the replica",
)

log_file = click.option(
    "-l",
    "--log_file",
    type=click.Path(path_type=pathlib.Path),
    default=None,
    required=False,
    help="path to log file",
)


@base_command.command("NS")
@runcard_path
@fit_card
@log_file
def nested_sampling(runcard_path, fit_card, log_file):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        setup_console(log_file)
        print_banner()
        log.console.log("Running : Nested Sampling Fit ")
        runner = Runner.from_file(runcard_path, fit_card)
    else:
        runner = None

    runner = comm.bcast(runner, root=0)
    runner.run_analysis("NS")


@base_command.command("MC")
@runcard_path
@fit_card
@n_replica
@log_file
def monte_carlo_fit(runcard_path, fit_card, n_replica, log_file):
    setup_console(log_file)
    print_banner()
    log.console.log("Running : MonteCarlo Fit")
    runner = Runner.from_file(runcard_path, fit_card, n_replica)
    runner.run_analysis("MC")


@base_command.command("PF")
@runcard_path
@fit_card
@n_replica
@click.option(
    "-c",
    "--clean_rep",
    is_flag=True,
    default=False,
    required=False,
    help="remove the replica file",
)
def post_fit(runcard_path, fit_card, n_replica, clean_rep):
    postfit = Postfit.from_file(runcard_path, fit_card)
    postfit.save(n_replica)
    if clean_rep:
        postfit.clean()


@base_command.command("R")
@runcard_path
@fit_card
def report(runcard_path, fit_card):
    run_report(runcard_path, fit_card)
