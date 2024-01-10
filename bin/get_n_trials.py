import optuna
import typer
import sys

def get_optuna_storage(optuna_journal_fname):
    storage = optuna.storages.JournalStorage(
        optuna.storages.JournalFileStorage(optuna_journal_fname),
    )
    return storage

def get_number_of_trials(storage, study_name):
    study_id = storage.get_study_id_from_name(study_name)
    return storage.get_n_trials(study_id)

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
