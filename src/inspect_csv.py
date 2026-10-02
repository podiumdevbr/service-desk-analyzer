from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURAÇÃO
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CSV_PATH = PROJECT_ROOT / "data" / "raw" / "conversas.csv"


# ============================================================
# FUNÇÕES
# ============================================================

def load_csv(csv_path: Path) -> pd.DataFrame:
    """
    Carrega o CSV tentando primeiro UTF-8 e, caso necessário,
    utiliza latin-1 como alternativa.
    """

    encodings = ["utf-8-sig", "utf-8", "latin-1"]

    last_error = None

    for encoding in encodings:
        try:
            df = pd.read_csv(
                csv_path,
                encoding=encoding,
                low_memory=False,
            )

            print(f"CSV carregado com encoding: {encoding}")
            return df

        except UnicodeDecodeError as error:
            last_error = error

    raise RuntimeError(
        "Não foi possível identificar o encoding do arquivo CSV."
    ) from last_error


def show_basic_information(df: pd.DataFrame) -> None:
    """Exibe informações básicas sobre o CSV."""

    print("\n" + "=" * 70)
    print("INFORMAÇÕES DO CSV")
    print("=" * 70)

    print(f"Quantidade de linhas: {len(df):,}")
    print(f"Quantidade de colunas: {len(df.columns)}")

    print("\nColunas:")
    for index, column in enumerate(df.columns, start=1):
        print(f"  {index:02d}. {column}")


def show_missing_values(df: pd.DataFrame) -> None:
    """Exibe quantidade de valores ausentes por coluna."""

    print("\n" + "=" * 70)
    print("VALORES AUSENTES")
    print("=" * 70)

    missing = df.isna().sum()

    for column, quantity in missing.items():
        print(f"{column}: {quantity:,}")


def show_protocol_information(df: pd.DataFrame) -> None:
    """Exibe informações sobre protocolos."""

    print("\n" + "=" * 70)
    print("PROTOCOLOS")
    print("=" * 70)

    if "Protocolo" not in df.columns:
        print("Coluna 'Protocolo' não encontrada.")
        return

    protocols = df["Protocolo"].dropna().astype(str)

    print(f"Protocolos distintos: {protocols.nunique():,}")

    print("\nMensagens por protocolo:")
    messages_per_protocol = protocols.value_counts()

    print(messages_per_protocol.describe().to_string())


def show_origin_information(df: pd.DataFrame) -> None:
    """Exibe distribuição da origem das mensagens."""

    print("\n" + "=" * 70)
    print("ORIGEM DAS MENSAGENS")
    print("=" * 70)

    if "Origem" not in df.columns:
        print("Coluna 'Origem' não encontrada.")
        return

    counts = (
        df["Origem"]
        .fillna("[VAZIO]")
        .astype(str)
        .value_counts()
    )

    print(counts.to_string())


def show_agent_information(df: pd.DataFrame) -> None:
    """Exibe agentes encontrados."""

    print("\n" + "=" * 70)
    print("AGENTES")
    print("=" * 70)

    if "Agente" not in df.columns:
        print("Coluna 'Agente' não encontrada.")
        return

    agents = (
        df["Agente"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    agents = agents[agents != ""]

    print(f"Agentes distintos: {agents.nunique():,}")

    for agent, quantity in agents.value_counts().items():
        print(f"{agent}: {quantity:,} mensagens")


def show_status_information(df: pd.DataFrame) -> None:
    """Exibe os valores encontrados em Status do atendimento."""

    print("\n" + "=" * 70)
    print("STATUS DO ATENDIMENTO")
    print("=" * 70)

    column = "Status do atendimento"

    if column not in df.columns:
        print(f"Coluna '{column}' não encontrada.")
        return

    counts = (
        df[column]
        .fillna("[VAZIO]")
        .astype(str)
        .value_counts()
    )

    print(counts.to_string())


def main() -> None:
    """Ponto de entrada do programa."""

    print("=" * 70)
    print("SERVICE DESK ANALYZER")
    print("INSPEÇÃO INICIAL DO CSV")
    print("=" * 70)

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado:\n{CSV_PATH}"
        )

    df = load_csv(CSV_PATH)

    show_basic_information(df)
    show_missing_values(df)
    show_protocol_information(df)
    show_origin_information(df)
    show_agent_information(df)
    show_status_information(df)

    print("\n" + "=" * 70)
    print("INSPEÇÃO CONCLUÍDA")
    print("=" * 70)


if __name__ == "__main__":
    main()