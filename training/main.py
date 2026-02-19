from config import Config
from fraud_detection_system import FraudDetectionSystem
from pathlib import Path

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  STANDALONE REŽIM - Testiranje sistema sa Warm-Start")
    print("=" * 70 + "\n")

    # Kreiraj custom config ako želiš (ili koristi default)
    custom_config = Config()
    custom_config.SAMPLE_FRACTION = 0.2  # Koristi samo 20% podataka za brže testiranje
    custom_config.USE_BALANCING = True
    custom_config.DEFAULT_BATCH_SIZE = 500

    # Kreiraj sistem
    system = FraudDetectionSystem(
        data_path=Path(__file__).parent.parent / 'data' / 'creditcard.csv',
        sample_fraction=custom_config.SAMPLE_FRACTION,
        use_balancing=custom_config.USE_BALANCING,
        config=custom_config
    )

    # Pokreni kompletan pipeline
    report = system.run_complete_pipeline(
        batch_size=500,
        streaming_delay=0,  # Bez pauze za brže izvršavanje
        save_results=True,
        warmup_samples=2000  # ✅ ARF će prvo naučiti 2000 primera iz RF-a
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