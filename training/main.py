from config import Config
from fraud_detection_system import FraudDetectionSystem
from pathlib import Path
import sys

def print_menu():
    print("  Test Selection")
    print("  1 - Default")
    print("  2 - Balanced")
    print("  3 - Fast")
    print("  4 - Custom")
    print("  5 - Exit")

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

def editor(base_config):
    # Napravi kopiju trenutne konfiguracije (da bismo mogli odbaciti promjene)
    new_config = Config()
    for attr in dir(base_config):
        if attr.isupper() and not callable(getattr(base_config, attr)):
            setattr(new_config, attr, getattr(base_config, attr))

    # Svi parametri koji se mogu uređivati (velika slova)
    params = sorted([attr for attr in dir(new_config) if attr.isupper() and not callable(getattr(new_config, attr))])

    while True:
        print("\n--- INTERAKTIVNO PODEŠAVANJE KONFIGURACIJE ---")
        print("Odaberite parametar za promjenu (unos broja):\n")

        for i, param in enumerate(params, 1):
            print(f"  {i}. {param} = {getattr(new_config, param)}")

        print(f"\n  {len(params) + 1}. 🚀 Pokreni test sa trenutnim podešavanjima")
        print(f"  {len(params) + 2}. ↩️  Nazad na glavni meni (odbaci promjene)")

        try:
            choice = int(input("\nVaš izbor: ").strip())
        except ValueError:
            print("Molimo unesite broj.")
            continue

        if 1 <= choice <= len(params):
            # Odabran parametar
            param = params[choice - 1]
            current_value = getattr(new_config, param)
            print(f"\nTrenutna vrijednost '{param}': {current_value}")
            new_input = input("Unesite novu vrijednost (Enter za odustajanje): ").strip()
            if new_input == "":
                print("Promjena otkazana.")
                continue

            # Konverzija tipa
            try:
                if isinstance(current_value, bool):
                    converted = new_input.lower() in ['da', 'true', '1', 'yes', 'y']
                elif isinstance(current_value, int):
                    converted = int(new_input)
                elif isinstance(current_value, float):
                    converted = float(new_input)
                elif isinstance(current_value, str):
                    converted = new_input
                else:
                    converted = new_input  # fallback
            except ValueError:
                print(f"Greška: '{new_input}' nije odgovarajućeg tipa ({type(current_value).__name__}).")
                continue

            setattr(new_config, param, converted)
            print(f"✅ Parametar '{param}' postavljen na {converted}")

        elif choice == len(params) + 1:
            # Pokreni test
            print("\nZavršeno uređivanje. Pokrećem test...")
            return new_config

        elif choice == len(params) + 2:
            # Nazad bez promjena
            print("\nOdustajem od promjena. Vraćam se na glavni meni.")
            return None

        else:
            print("Nepostojeća opcija.")

if __name__ == '__main__':
    while True:
        print_menu()
        choice = input().strip()

        if choice == '5':
            sys.exit(0)

        if choice not in ['1', '2', '3', '4']:
            print("Invalid selection")
            continue

        # Kreiraj osnovnu konfiguraciju
        custom_config = Config()

        # Primijeni odabranu opciju
        if choice == '2':
            custom_config = balanced(custom_config)
        elif choice == '3':
            custom_config = fast(custom_config)
        elif choice == '4':
            modified_config = custom_config = editor(custom_config)
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