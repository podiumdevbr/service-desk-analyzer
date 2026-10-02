from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURAÇÃO
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "conversas.csv"
)


REQUIRED_COLUMNS = [
    "Contato",
    "Identificador",
    "Protocolo",
    "Canal",
    "Início do atendimento",
    "Data da mensagem",
    "Origem",
    "Agente",
    "Plataforma",
    "Mensagem",
    "ID Mensagem",
    "ID Contexto",
    "Status do atendimento",
]


# ============================================================
# CARREGAMENTO
# ============================================================

def load_csv(csv_path: Path = DEFAULT_CSV_PATH) -> pd.DataFrame:
    """
    Carrega o CSV utilizando diferentes possibilidades de encoding.
    """

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Arquivo CSV não encontrado:\n{csv_path}"
        )

    encodings = [
        "utf-8-sig",
        "utf-8",
        "latin-1",
    ]

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


# ============================================================
# VALIDAÇÃO DAS COLUNAS
# ============================================================

def validate_columns(df: pd.DataFrame) -> None:
    """
    Verifica se todas as colunas esperadas estão presentes.
    """

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        missing = "\n".join(
            f"  - {column}"
            for column in missing_columns
        )

        raise ValueError(
            "O CSV não possui todas as colunas obrigatórias.\n\n"
            f"Colunas ausentes:\n{missing}"
        )


# ============================================================
# PADRONIZAÇÃO
# ============================================================

def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Padroniza os principais tipos de dados utilizados pelo projeto.
    """

    df = df.copy()

    # --------------------------------------------------------
    # Colunas textuais
    # --------------------------------------------------------

    text_columns = [
        "Contato",
        "Identificador",
        "Protocolo",
        "Canal",
        "Origem",
        "Agente",
        "Plataforma",
        "Mensagem",
        "ID Mensagem",
        "ID Contexto",
        "Status do atendimento",
    ]

    for column in text_columns:
        if column in df.columns:
            df[column] = (
                df[column]
                .fillna("")
                .astype(str)
                .str.strip()
            )

    # --------------------------------------------------------
    # Datas
    #
    # O CSV exportado pelo Service Desk utiliza o padrão:
    #
    # DD/MM/AAAA HH:MM:SS
    #
    # O formato é informado explicitamente para evitar que o
    # pandas faça uma interpretação automática das datas.
    # --------------------------------------------------------

    date_format = "%d/%m/%Y %H:%M:%S"

    df["Início do atendimento"] = pd.to_datetime(
        df["Início do atendimento"],
        format=date_format,
        errors="coerce",
    )

    df["Data da mensagem"] = pd.to_datetime(
        df["Data da mensagem"],
        format=date_format,
        errors="coerce",
    )

    # --------------------------------------------------------
    # Ordenação
    #
    # Cada protocolo continua sendo uma unidade independente.
    # A ordenação é feita primeiro pelo protocolo e depois pela
    # data da mensagem.
    # --------------------------------------------------------

    df = df.sort_values(
        by=[
            "Protocolo",
            "Data da mensagem",
        ],
        kind="stable",
    ).reset_index(drop=True)

    return df


# ============================================================
# VALIDAÇÃO DOS DADOS
# ============================================================

def validate_data(df: pd.DataFrame) -> None:
    """
    Verifica problemas básicos que poderiam comprometer a análise.
    """

    if df.empty:
        raise ValueError("O CSV está vazio.")

    # --------------------------------------------------------
    # Protocolos vazios
    # --------------------------------------------------------

    empty_protocols = (
        df["Protocolo"]
        .isna()
        .sum()
    )

    if empty_protocols > 0:
        raise ValueError(
            "Foram encontrados registros sem Protocolo: "
            f"{empty_protocols:,}"
        )

    empty_protocols = (
        df["Protocolo"]
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    if empty_protocols > 0:
        raise ValueError(
            "Foram encontrados registros com Protocolo vazio: "
            f"{empty_protocols:,}"
        )

    # --------------------------------------------------------
    # Datas inválidas
    # --------------------------------------------------------

    invalid_message_dates = (
        df["Data da mensagem"]
        .isna()
        .sum()
    )

    if invalid_message_dates > 0:
        print(
            "AVISO: "
            f"{invalid_message_dates:,} mensagens possuem "
            "Data da mensagem inválida."
        )

    invalid_start_dates = (
        df["Início do atendimento"]
        .isna()
        .sum()
    )

    if invalid_start_dates > 0:
        print(
            "AVISO: "
            f"{invalid_start_dates:,} registros possuem "
            "Início do atendimento inválido."
        )


# ============================================================
# FUNÇÃO PRINCIPAL DE PREPARAÇÃO
# ============================================================

def prepare_conversations(
    csv_path: Path = DEFAULT_CSV_PATH,
) -> pd.DataFrame:
    """
    Carrega, valida e prepara o CSV para as próximas etapas.
    """

    df = load_csv(csv_path)

    validate_columns(df)

    df = normalize_dataframe(df)

    validate_data(df)

    return df


# ============================================================
# RESUMO
# ============================================================

def show_summary(df: pd.DataFrame) -> None:
    """
    Exibe um resumo do conjunto de dados preparado.
    """

    print("\n" + "=" * 70)
    print("RESUMO DOS DADOS")
    print("=" * 70)

    print(f"Mensagens: {len(df):,}")

    print(
        "Protocolos: "
        f"{df['Protocolo'].nunique():,}"
    )

    print(
        "Identificadores: "
        f"{df['Identificador'].nunique():,}"
    )

    if not df["Data da mensagem"].isna().all():
        first_date = df["Data da mensagem"].min()
        last_date = df["Data da mensagem"].max()

        print(
            "Período: "
            f"{first_date:%d/%m/%Y %H:%M:%S} "
            "até "
            f"{last_date:%d/%m/%Y %H:%M:%S}"
        )

    print(
        "Agentes: "
        f"{df['Agente'].replace('', pd.NA).nunique():,}"
    )

    print("=" * 70)


# ============================================================
# EXECUÇÃO DIRETA
# ============================================================

def main() -> None:
    """
    Permite testar o módulo diretamente pelo terminal.
    """

    print("=" * 70)
    print("SERVICE DESK ANALYZER")
    print("PREPARAÇÃO DOS DADOS")
    print("=" * 70)

    df = prepare_conversations()

    show_summary(df)

    print("\nPreparação concluída com sucesso.")


if __name__ == "__main__":
    main()