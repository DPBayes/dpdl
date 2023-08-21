import optuna
import typer

from rich.pretty import pprint

def get_optuna_storage(optuna_journal_fname):
    storage = optuna.storages.JournalStorage(
        optuna.storages.JournalFileStorage(optuna_journal_fname),
    )

    return storage

def print_best_trials(storage, verbose=False):
    studies = storage.get_all_studies()
    for study in studies:
        print('----------------------------------------------------------------')
        print(f'Study: {study.study_name}')
        study_id = storage.get_study_id_from_name(study.study_name)
        trial = storage.get_best_trial(study_id)

        print(f'- Best trial:')
        print_trial(trial, verbose=verbose)

def print_trial(trial, verbose=False):
    if verbose:
        pprint(trial)
    else:
        print(f' - Number         : {trial.number}')
        print(f' - State          : {trial.state}')
        print(f' - Objective value: {trial.values[0]}')
        print(f' - Params         : {trial.params}')


def main(optuna_journal: str = '../optuna-journal.log', verbose: bool = False):
    """
    Reporting tool for Differentially Private Deep Learning experiments.

    Parses an Optuna journal log file and reports the best trials with their parameters.

    --optuna_journal is an Optuna journal filename

    --verbose flag will print all the parameters instead of just the selected ones
    """
    storage = get_optuna_storage(optuna_journal)
    print_best_trials(storage, verbose=verbose)

if __name__ == '__main__':
    typer.run(main)
