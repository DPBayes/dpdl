import optuna
import typer
import sys
import os

def get_optuna_storage(optuna_journal_fname):
    if not os.path.exists(optuna_journal_fname):
        print(f'Error: Optuna journal file "{optuna_journal_fname}" not found.')
        sys.exit(1)

    storage = optuna.storages.JournalStorage(
        optuna.storages.JournalFileStorage(optuna_journal_fname),
    )
    return storage

def get_number_of_trials(storage, study_name):
    try:
        study_id = storage.get_study_id_from_name(study_name)
        return storage.get_n_trials(study_id)
    except KeyError:
        print(f'Error: Study "{study_name}" does not exist.')
        sys.exit(1)
    except Exception as e:
        print(f'An unexpected error occurred: {e}')
        sys.exit(1)

def main(optuna_journal: str = '../optuna.journal', study_name: str = ''):
    """
    Reports the number of trials in an Optuna study.
    """

    if not study_name:
        print(f'Usage: python get_n_trials.py --study-name <STUDY NAME>')
        sys.exit(1)

    storage = get_optuna_storage(optuna_journal)
    num_trials = get_number_of_trials(storage, study_name)
    print(str(num_trials))

if __name__ == '__main__':
    typer.run(main)
