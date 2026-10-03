from pathlib import Path
import pandas as pd

from conversations import prepare_conversations
from protocols import build_protocol_summary, prepare_messages


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "protocolos_analise.csv"
)


# ============================================================
# CONVERSÃO DE TEMPOS
# ============================================================

def timedelta_to_minutes(series: pd.Series) -> pd.Series:
    """Converte uma série de Timedelta para minutos."""
    return (
        series
        .dt.total_seconds()
        .div(60)
    )


# ============================================================
# MÉTRICAS DERIVADAS POR PROTOCOLO
# ============================================================

def calculate_protocol_metrics(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Cria a camada analítica derivada a partir do resumo de protocolos.

    A unidade de análise continua sendo exclusivamente o Protocolo.
    """

    if summary.empty:
        return summary.copy()

    result = summary.copy()

    # --------------------------------------------------------
    # Durações
    # --------------------------------------------------------

    result["Duração bruta minutos"] = timedelta_to_minutes(
        result["Duração bruta protocolo"]
    )

    result["Duração humana detectada minutos"] = timedelta_to_minutes(
        result["Duração atendimento humano detectado"]
    )

    result["Tempo até última interação minutos"] = timedelta_to_minutes(
        result["Tempo do início humano até última interação"]
    )

    result["Tempo efetivo humano minutos"] = timedelta_to_minutes(
        result["Tempo efetivo atendimento humano"]
    )

    result["Tempo retomada entre dias minutos"] = timedelta_to_minutes(
        result["Tempo retomada entre dias"]
    )

    result["Tempo expediente normal minutos"] = timedelta_to_minutes(
        result["Tempo dentro do expediente"]
    )

    result["Tempo extensão 19h-20h05 minutos"] = timedelta_to_minutes(
        result["Tempo extensão 19h-20h05"]
    )

    result["Tempo fora expediente minutos"] = timedelta_to_minutes(
        result["Tempo fora do expediente"]
    )

    # --------------------------------------------------------
    # Indicadores de interação
    # --------------------------------------------------------

    result["Taxa mensagens contato"] = (
        result["Mensagens contato"]
        .div(result["Quantidade mensagens"].replace(0, pd.NA))
    )

    result["Taxa mensagens agente"] = (
        result["Mensagens agente"]
        .div(result["Quantidade mensagens"].replace(0, pd.NA))
    )

    result["Possui áudio"] = (
        result["Quantidade áudios"] > 0
    )

    result["Possui múltiplos agentes"] = (
        result["Quantidade agentes"] > 1
    )

    result["Possui retomada"] = (
        result["Quantidade retomadas outro dia"] > 0
    )

    result["Possui atividade após encerramento"] = (
        result["Houve interação após encerramento"]
    )

    result["Possui atividade fora do expediente"] = (
        result["Tempo fora expediente minutos"] > 0
    )

    result["Possui atividade na extensão"] = (
        result["Tempo extensão 19h-20h05 minutos"] > 0
    )

    # --------------------------------------------------------
    # Classificação operacional
    # --------------------------------------------------------

    def classify_service_pattern(row: pd.Series) -> str:
        if row["Possui retomada"]:
            return "RETOMADO_OUTRO_DIA"

        if row["Possui atividade na extensão"]:
            return "ATENDIMENTO_COM_EXTENSAO"

        if row["Possui atividade fora do expediente"]:
            return "ATENDIMENTO_FORA_EXPEDIENTE"

        return "ATENDIMENTO_INTRADIA"

    result["Padrão temporal atendimento"] = result.apply(
        classify_service_pattern,
        axis=1,
    )

    return result


# ============================================================
# RESUMO GERAL DAS MÉTRICAS
# ============================================================

def calculate_overall_metrics(
    metrics_df: pd.DataFrame,
) -> dict:
    """Calcula indicadores agregados do conjunto de protocolos."""

    if metrics_df.empty:
        return {
            "protocolos": 0,
        }

    human = metrics_df[
        metrics_df["Houve atendimento humano"]
    ].copy()

    return {
        "protocolos": int(len(metrics_df)),
        "protocolos_com_atendimento_humano": int(len(human)),
        "protocolos_com_retomada": int(
            human["Possui retomada"].sum()
        ),
        "protocolos_com_audio": int(
            human["Possui áudio"].sum()
        ),
        "protocolos_com_multiplos_agentes": int(
            human["Possui múltiplos agentes"].sum()
        ),
        "protocolos_com_atividade_pos_encerramento": int(
            human["Possui atividade após encerramento"].sum()
        ),
        "protocolos_com_extensao": int(
            human["Possui atividade na extensão"].sum()
        ),
        "protocolos_com_fora_expediente": int(
            human["Possui atividade fora do expediente"].sum()
        ),
        "media_duracao_bruta_minutos": float(
            metrics_df["Duração bruta minutos"].dropna().mean()
        ),
        "mediana_duracao_bruta_minutos": float(
            metrics_df["Duração bruta minutos"].dropna().median()
        ),
        "media_tempo_efetivo_humano_minutos": float(
            human["Tempo efetivo humano minutos"].dropna().mean()
        ),
        "mediana_tempo_efetivo_humano_minutos": float(
            human["Tempo efetivo humano minutos"].dropna().median()
        ),
    }


# ============================================================
# DISTRIBUIÇÕES
# ============================================================

def calculate_distribution_tables(
    metrics_df: pd.DataFrame,
) -> dict:
    """Produz tabelas agregadas para as próximas análises."""

    return {
        "status_humano": (
            metrics_df[
                "Status atendimento humano"
            ]
            .value_counts()
            .rename_axis("Status")
            .reset_index(name="Protocolos")
        ),
        "padrao_temporal": (
            metrics_df[
                "Padrão temporal atendimento"
            ]
            .value_counts()
            .rename_axis("Padrão")
            .reset_index(name="Protocolos")
        ),
        "tipo_encerramento": (
            metrics_df[
                "Tipo encerramento detectado"
            ]
            .value_counts()
            .rename_axis("Tipo encerramento")
            .reset_index(name="Protocolos")
        ),
        "agentes": (
            metrics_df[
                metrics_df["Agentes"].astype(str).str.strip() != ""
            ]["Agentes"]
            .value_counts()
            .rename_axis("Agentes")
            .reset_index(name="Protocolos")
        ),
    }


# ============================================================
# EXECUÇÃO
# ============================================================

def build_metrics(
    csv_path=None,
) -> tuple[pd.DataFrame, dict, dict]:
    """Executa preparação, consolidação e cálculo das métricas."""

    if csv_path is None:
        csv_path = (
            PROJECT_ROOT
            / "data"
            / "raw"
            / "conversas.csv"
        )

    df = prepare_conversations(csv_path)
    prepared = prepare_messages(df)
    summary = build_protocol_summary(prepared)
    metrics_df = calculate_protocol_metrics(summary)

    overall = calculate_overall_metrics(metrics_df)
    distributions = calculate_distribution_tables(metrics_df)

    return metrics_df, overall, distributions


def save_metrics(
    metrics_df: pd.DataFrame,
    output_path=DEFAULT_OUTPUT_PATH,
) -> Path:
    """Salva a tabela analítica por protocolo em CSV."""

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    return output_path


def show_metrics_report(
    metrics_df: pd.DataFrame,
    overall: dict,
) -> None:
    """Exibe o resumo da camada de métricas."""

    print("\n" + "=" * 70)
    print("CAMADA DE MÉTRICAS")
    print("=" * 70)

    print(
        f"Protocolos analisados: "
        f"{overall['protocolos']:,}"
    )

    print(
        f"Com atendimento humano: "
        f"{overall['protocolos_com_atendimento_humano']:,}"
    )

    print(
        f"Com retomada em outro dia: "
        f"{overall['protocolos_com_retomada']:,}"
    )

    print(
        f"Com áudio: "
        f"{overall['protocolos_com_audio']:,}"
    )

    print(
        f"Com múltiplos agentes: "
        f"{overall['protocolos_com_multiplos_agentes']:,}"
    )

    print(
        f"Com atividade após encerramento: "
        f"{overall['protocolos_com_atividade_pos_encerramento']:,}"
    )

    print(
        f"Com extensão 19h-20h05: "
        f"{overall['protocolos_com_extensao']:,}"
    )

    print(
        f"Com atividade fora do expediente: "
        f"{overall['protocolos_com_fora_expediente']:,}"
    )

    print("\nDURAÇÃO BRUTA")

    print(
        f"  Média: "
        f"{overall['media_duracao_bruta_minutos']:.2f} minutos"
    )

    print(
        f"  Mediana: "
        f"{overall['mediana_duracao_bruta_minutos']:.2f} minutos"
    )

    print("\nTEMPO EFETIVO HUMANO")

    print(
        f"  Média: "
        f"{overall['media_tempo_efetivo_humano_minutos']:.2f} minutos"
    )

    print(
        f"  Mediana: "
        f"{overall['mediana_tempo_efetivo_humano_minutos']:.2f} minutos"
    )


def main() -> None:
    """Executa a camada de métricas."""

    metrics_df, overall, _ = build_metrics()

    output_path = save_metrics(metrics_df)

    show_metrics_report(
        metrics_df,
        overall,
    )

    print("\n" + "=" * 70)
    print("MÉTRICAS CONCLUÍDAS")
    print("=" * 70)

    print(
        f"Arquivo gerado: {output_path}"
    )


if __name__ == "__main__":
    main()
