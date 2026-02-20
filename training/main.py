from config import Config
from fraud_detection_system import FraudDetectionSystem
from pathlib import Path
import sys

def print_menu():
    print("\nTest Selection")
    print("1.  Default")
    print("2. Balanced")
    print("3. Fast")
    print("4. Custom")
    print("5. Exit")

def balanced(config):
    config.SAMPLE_FRACTION = 1.0
    config.USE_BALANCING = True
    config.SMOTE_SAMPLING_STRATEGY = 0.4
    config.UNDERSAMPLING_STRATEGY = 0.6
    config.RF_N_ESTIMATORS = 200
    config.RF_MAX_DEPTH = 30
    config.ARF_N_MODELS = 20
    config.ARF_LAMBDA = 10
    return config

def fast(config):
    config.SAMPLE_FRACTION = 0.1
    config.USE_BALANCING = False
    config.RF_N_ESTIMATORS = 20
    config.RF_MAX_DEPTH = 10
    config.ARF_N_MODELS = 3
    config.ARF_LAMBDA = 3
    return config

def custom():
    new_config = Config()

    # ostavljanje samo zeljenih parametara za izmenu
    excluded_params = {'DEFAULT_BATCH_SIZE', 'RANDOM_SEED', 'RF_N_JOBS'}
    params = []
    for attr in dir(new_config):
        if attr in excluded_params:
            continue
        value = getattr(new_config, attr)
        if type(value) in (int, float) and not isinstance(value, bool):
            params.append(attr)

    while True:
        print("\nChange parameter\n")

        for i, param in enumerate(params, 1):
            print(f"{i}. {param} = {getattr(new_config, param)}")

        print(f"\n{len(params) + 1}. Apply and start")
        print(f"{len(params) + 2}. Drop changes and exit")

        try:
            choice = int(input().strip())
        except ValueError:
            print("Invalid selection")
            continue

        # Opcija za promjenu parametra
        if 1 <= choice <= len(params):
            param = params[choice - 1]
            current_value = getattr(new_config, param)
            print(f"\nCurrent value '{param}': {current_value}")
            new_input = input("Change value or leave blank to cancel: ").strip()
            if new_input == "":
                continue

            # konverzija u odgovarajuci tip
            try:
                if isinstance(current_value, int):
                    converted = int(new_input)
                elif isinstance(current_value, float):
                    converted = float(new_input)
                else:
                    converted = new_input
            except ValueError:
                print("Invalid value\n")
                continue

            setattr(new_config, param, converted)

        elif choice == len(params) + 1:
            return new_config

        elif choice == len(params) + 2:
            return None

        else:
            print("Invalid selection")

if __name__ == '__main__':
    while True:
        print_menu()
        choice = input().strip()

        if choice == '5':
            sys.exit(0)

        if choice not in ['1', '2', '3', '4']:
            print("Invalid selection")
            continue

        custom_config = Config()

        if choice == '2':
            custom_config = balanced(custom_config)
        elif choice == '3':
            custom_config = fast(custom_config)
        elif choice == '4':
            modified_config = custom()
            if modified_config is None:
                continue
            else:
                custom_config = modified_config

        system = FraudDetectionSystem(
            data_path=Path(__file__).parent.parent / 'data' / 'creditcard.csv',
            config=custom_config
        )

        report = system.run_complete_pipeline(
            streaming_delay=0,  # ako treba za simulaciju cekanja na sledecu transakciju
            save_results=True,
            warmup_samples=2000
        )

        print("\n" + "=" * 70)
        print("  FINALNI IZVEŠTAJ")
        print("=" * 70)
        print(f"\nVreme izvršavanja: {report['elapsed_time_seconds']:.2f}s")
        print(f"Inicijalni RF F1-Score: {report['initial_model_results']['f1'] * 100:.2f}%")

        if report['streaming_summary']:
            totals = report['streaming_summary']['totals']
            print(f"\nUkupno detektovano prevara: {totals['frauds_detected']}/{totals['frauds_encountered']}")
            print(f"Overall Detection Rate: {totals['overall_detection_rate'] * 100:.2f}%")

            print(f"\n💡 ARF je počeo sa {report['configuration']['warmup_samples']} primera znanja iz RF-a!")