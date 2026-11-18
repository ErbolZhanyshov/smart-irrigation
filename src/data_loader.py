"""
Smart Irrigation System - Data Loader Module
Merkezi veri yükleme, doğrulama ve ön işleme fonksiyonları.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np


# Merkezi beklenen sütunlar ve veri tipleri
EXPECTED_COLUMNS = [
    "date",
    "Soil Moisture Resistance (Ohm)",
    "Ambient Temperature (deg. C)",
    "Atmospheric Humidity (%)",
    "DC Water pump Status",
]

EXPECTED_DTYPES = {
    "Soil Moisture Resistance (Ohm)": "int64",
    "Ambient Temperature (deg. C)": "float64",
    "Atmospheric Humidity (%)": "int64",
    "DC Water pump Status": "int64",
}

DATE_FORMAT = "%d-%m-%Y %H:%M"


def get_project_root() -> Path:
    """Proje kök dizinini güvenli ve dinamik olarak döner."""
    return Path(__file__).resolve().parent.parent


def get_default_dataset_path() -> Path:
    """Yeni birincil veri setinin (project_dataset.csv) varsayılan yolunu döner."""
    return get_project_root() / "data" / "raw" / "project_dataset.csv"


def load_data(
    filepath: Optional[Path] = None,
    parse_dates: bool = True,
    sort_chronologically: bool = True,
) -> pd.DataFrame:
    """
    project_dataset.csv dosyasını yükler, sütunları ve tipleri doğrular.

    Parametreler:
        filepath: Veri seti dosya yolu (None ise varsayılan project_dataset.csv kullanılır).
        parse_dates: Tarih sütununu datetime formatına çevirip çevirmeyeceği.
        sort_chronologically: Veriyi kronolojik olarak sıralayıp sıralamayacağı.

    Dönen Değer:
        pd.DataFrame: Doğrulanmış veri çerçevesi.
    """
    path = Path(filepath) if filepath else get_default_dataset_path()

    if not path.exists():
        raise FileNotFoundError(
            f"Veri seti dosyası bulunamadı: {path.resolve()}\n"
            "Lütfen 'data/raw/project_dataset.csv' dosyasının mevcut olduğunu doğrulayın."
        )

    # CSV okuma
    df = pd.read_csv(path)

    # 1. Sütun isimleri doğrulama
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Veri setinde beklenen sütunlar eksik: {missing_cols}")

    # Sütun sırasını standartlaştır
    df = df[EXPECTED_COLUMNS].copy()

    # 2. Tarih sütununu datetime tipine dönüştürme
    if parse_dates:
        try:
            df["date"] = pd.to_datetime(df["date"], format=DATE_FORMAT)
        except Exception as e:
            raise ValueError(
                f"Tarih sütunu beklenen formatta ('{DATE_FORMAT}') ayrıştırılamadı: {e}"
            )

        # 3. Kronolojik sıra kontrolü ve gerekiyorsa sıralama
        is_sorted = df["date"].is_monotonic_increasing
        if not is_sorted and sort_chronologically:
            df = df.sort_values("date").reset_index(drop=True)

    # 4. Sayısal veri tiplerini güvenli şekilde dönüştürme
    for col, dtype in EXPECTED_DTYPES.items():
        try:
            df[col] = df[col].astype(dtype)
        except Exception as e:
            raise TypeError(f"'{col}' sütunu '{dtype}' tipine dönüştürülemedi: {e}")

    return df


def inspect_temperature_anomalies(df: pd.DataFrame, threshold: float = 10.0) -> Dict[str, Any]:
    """
    Sıcaklık sütunundaki şüpheli aşırı düşük değerleri (ör. 3.4 °C) tespit eder.
    Ham veriyi değiştirmez veya silmez; yalnızca akademik ve analitik amaçla raporlar.
    """
    temp_col = "Ambient Temperature (deg. C)"
    if temp_col not in df.columns:
        return {"error": f"'{temp_col}' sütunu bulunamadı."}

    suspicious = df[df[temp_col] < threshold].copy()

    records = []
    for idx, row in suspicious.iterrows():
        # Çevresindeki satırları (öncesi ve sonrası) inceleyerek ani düşüşü tespit et
        prev_idx = max(0, idx - 1)
        next_idx = min(len(df) - 1, idx + 1)
        prev_temp = df.loc[prev_idx, temp_col]
        next_temp = df.loc[next_idx, temp_col]

        records.append({
            "index": int(idx),
            "date": str(row["date"]),
            "temperature": float(row[temp_col]),
            "prev_temperature": float(prev_temp),
            "next_temperature": float(next_temp),
            "soil_resistance": float(row["Soil Moisture Resistance (Ohm)"]),
            "humidity": float(row["Atmospheric Humidity (%)"]),
            "pump_status": int(row["DC Water pump Status"]),
        })

    return {
        "threshold": threshold,
        "suspicious_count": len(suspicious),
        "suspicious_records": records,
        "note": (
            "Ham veriye dokunulmamıştır. Bu değerler (ör. 3.4 °C) ardışık 30 °C+ değerlerin "
            "arasında tekil 15 dakikalık periyotlarda görülmekte olup, olası sensör okuma "
            "veya veri yazım (ondalık kayması: 33.4 -> 3.4) hatasına işaret etmektedir."
        ),
    }


def validate_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Veri setini bütünlük, eksik değerler, tekrarlar ve sınır kontrolleri açısından doğrular.
    """
    validation_report = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "total_missing": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "is_chronological": bool(df["date"].is_monotonic_increasing) if "date" in df else None,
        "validations_passed": True,
        "issues": [],
    }

    # Eksik değer kontrolü
    if validation_report["total_missing"] > 0:
        validation_report["validations_passed"] = False
        validation_report["issues"].append(
            f"Toplam {validation_report['total_missing']} eksik değer bulundu."
        )

    # Tekrarlayan satır kontrolü
    if validation_report["duplicate_rows"] > 0:
        validation_report["validations_passed"] = False
        validation_report["issues"].append(
            f"Toplam {validation_report['duplicate_rows']} tekrarlayan satır bulundu."
        )

    # Pompa durumu kontrolü (yalnızca 0 ve 1 olmalı)
    pump_unique = set(df["DC Water pump Status"].unique())
    if not pump_unique.issubset({0, 1}):
        validation_report["validations_passed"] = False
        validation_report["issues"].append(
            f"Geçersiz pompa durumu değerleri tespit edildi: {pump_unique - {0, 1}}"
        )

    # Nem sınır kontrolü (%0 - %100)
    humidity_invalid = ((df["Atmospheric Humidity (%)"] < 0) | (df["Atmospheric Humidity (%)"] > 100)).sum()
    if humidity_invalid > 0:
        validation_report["validations_passed"] = False
        validation_report["issues"].append(
            f"{humidity_invalid} adet sınır dışı nem değeri bulundu."
        )

    # Toprak nem direnci kontrolü (> 0)
    resistance_invalid = (df["Soil Moisture Resistance (Ohm)"] <= 0).sum()
    if resistance_invalid > 0:
        validation_report["validations_passed"] = False
        validation_report["issues"].append(
            f"{resistance_invalid} adet sıfır veya negatif toprak direnci değeri bulundu."
        )

    # Şüpheli sıcaklık incelemesi
    temp_analysis = inspect_temperature_anomalies(df, threshold=10.0)
    validation_report["temperature_anomalies"] = temp_analysis

    return validation_report


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Veri setine ilişkin genel istatistikleri ve özet bilgileri döner.
    FastAPI veya EDA modülleri tarafından doğrudan kullanılabilir.
    """
    pump_counts = df["DC Water pump Status"].value_counts().to_dict()

    summary = {
        "total_records": len(df),
        "total_columns": len(df.columns),
        "columns": list(df.columns),
        "date_range": {
            "start": str(df["date"].min()),
            "end": str(df["date"].max()),
        },
        "pump_status": {
            "off": int(pump_counts.get(0, 0)),
            "on": int(pump_counts.get(1, 0)),
            "off_percentage": round(float(pump_counts.get(0, 0) / len(df) * 100), 2),
            "on_percentage": round(float(pump_counts.get(1, 0) / len(df) * 100), 2),
        },
        "numeric_stats": {
            col: {
                "mean": round(float(df[col].mean()), 2),
                "std": round(float(df[col].std()), 2),
                "min": round(float(df[col].min()), 2),
                "25%": round(float(df[col].quantile(0.25)), 2),
                "50%": round(float(df[col].median()), 2),
                "75%": round(float(df[col].quantile(0.75)), 2),
                "max": round(float(df[col].max()), 2),
            }
            for col in EXPECTED_COLUMNS
            if col != "date"
        },
    }
    return summary


if __name__ == "__main__":
    print("=" * 60)
    print("SMART IRRIGATION SYSTEM - DATA LOADER TEST")
    print("=" * 60)

    try:
        data = load_data()
        print(f"✓ Veri seti başarıyla yüklendi: {data.shape[0]} satır, {data.shape[1]} sütun")
        print(f"✓ Tarih sütun tipi: {data['date'].dtype}")
        print(f"✓ Tarih aralığı: {data['date'].min()} -> {data['date'].max()}")
        print(f"✓ Kronolojik sıralı mı: {data['date'].is_monotonic_increasing}")

        report = validate_dataset(data)
        print("\n--- DOĞRULAMA RAPORU ---")
        print(f"Doğrulama başarılı mı: {report['validations_passed']}")
        print(f"Eksik değer sayısı: {report['total_missing']}")
        print(f"Tekrarlayan satır sayısı: {report['duplicate_rows']}")
        if report["issues"]:
            print("Sorunlar:", report["issues"])
        else:
            print("✓ Herhangi bir veri bütünlüğü sorunu tespit edilmedi.")

        temp_info = report["temperature_anomalies"]
        print(f"\n--- SICAKLIK ŞÜPHELİ DEĞER İNCELEMESİ (< {temp_info['threshold']} °C) ---")
        print(f"Tespit edilen kayıt sayısı: {temp_info['suspicious_count']}")
        for rec in temp_info["suspicious_records"]:
            print(
                f"  - İndeks {rec['index']} | Tarih: {rec['date']} | Sıcaklık: {rec['temperature']} °C "
                f"(Önceki: {rec['prev_temperature']} °C, Sonraki: {rec['next_temperature']} °C)"
            )
        print(f"Açıklama: {temp_info['note']}")

        summary = get_dataset_summary(data)
        print("\n--- VERİ SETİ ÖZETİ ---")
        print(f"Pompa Kapalı (OFF - 0): {summary['pump_status']['off']} (%{summary['pump_status']['off_percentage']})")
        print(f"Pompa Açık (ON - 1): {summary['pump_status']['on']} (%{summary['pump_status']['on_percentage']})")

    except Exception as err:
        print(f"✗ HATA: {err}")

