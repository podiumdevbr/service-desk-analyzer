from pathlib import Path
import pandas as pd

from conversations import prepare_conversations

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


WORK_START_HOUR = 6
WORK_START_MINUTE = 30

WORK_END_HOUR = 19
WORK_END_MINUTE = 0

EXTENSION_END_HOUR = 20
EXTENSION_END_MINUTE = 5


# ============================================================
# EVENTOS TÉCNICOS CONHECIDOS
# ============================================================

TRANSFER_MESSAGE = (
    "Transferido automaticamente para equipe Atendimento SCRE"
)

EXPLICIT_CLOSURE_MESSAGE = (
    "Seu atendimento foi finalizado."
)

INACTIVITY_CLOSURE_TERMS = [
    "atendimento finalizado por inatividade",
]

EQUIVOCAL_CLOSURE_TERMS = [
    "atendimento foi finalizado por equivoco",
    "atendimento foi finalizado por equívoco",
    "atendimento finalizado por equivoco",
    "atendimento finalizado por equívoco",
    "seu atendimento foi finalizado por equivoco",
    "seu atendimento foi finalizado por equívoco",
]


# ============================================================
# ACEITE DO ATENDIMENTO HUMANO
# ============================================================

AGENT_ACCEPTANCE_TERMS = [
    "me chamo",
    "vou realizar",
    "seu atendimento",
]


# ============================================================
# TERMOS DE ENCERRAMENTO
# ============================================================

CLOSURE_TERMS = [
    "atendimento finalizado",
    "atendimento foi finalizado",
    "atendimento encerrado",
    "atendimento foi encerrado",
    "finalizado por inatividade",
    "encerrado por inatividade",
    "finalização do atendimento",
    "finalizacao do atendimento",
    "encerramento do atendimento",
]


CLOSURE_MESSAGE_TYPES = [
    "evento_encerramento",
    "evento_inatividade",
    "evento_encerramento_equivoco",
]


TECHNICAL_MESSAGE_TYPES = [
    "evento_transferencia",
    "evento_encerramento",
    "evento_inatividade",
    "evento_encerramento_equivoco",
    "evento_protocolo",
    "evento_avaliacao",
    "evento_aceite_atendimento_humano",
]


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def normalize_message(message: str) -> str:
    """
    Normaliza uma mensagem apenas para comparação.

    A mensagem original não é alterada.
    """

    if message is None:
        return ""

    return (
        str(message)
        .strip()
        .lower()
    )


def is_audio_message(message: str) -> bool:
    """
    Identifica mensagens que representam arquivos de áudio.
    """

    normalized = normalize_message(message)

    if not normalized:
        return False

    return (
        "arquivo de áudio" in normalized
        or "arquivo de audio" in normalized
    )


def is_transfer_message(message: str) -> bool:
    """
    Identifica o evento conhecido de transferência automática.
    """

    return (
        normalize_message(message)
        == normalize_message(TRANSFER_MESSAGE)
    )


def is_inactivity_closure(message: str) -> bool:
    """
    Identifica encerramento por inatividade.
    """

    normalized = normalize_message(message)

    if not normalized:
        return False

    return any(
        term.lower() in normalized
        for term in INACTIVITY_CLOSURE_TERMS
    )


def is_equivocal_closure(message: str) -> bool:
    """
    Identifica encerramentos registrados como equivocados.
    """

    normalized = normalize_message(message)

    if not normalized:
        return False

    return any(
        term.lower() in normalized
        for term in EQUIVOCAL_CLOSURE_TERMS
    )


def is_explicit_closure(message: str) -> bool:
    """
    Identifica encerramento normal do atendimento.

    A regra não exige igualdade exata, permitindo reconhecer
    mensagens de encerramento acompanhadas de HTML ou
    instruções de avaliação.
    """

    normalized = normalize_message(message)

    if not normalized:
        return False

    if is_inactivity_closure(message):
        return False

    if is_equivocal_closure(message):
        return False

    return (
        "seu atendimento foi finalizado" in normalized
        or "atendimento foi finalizado" in normalized
    )


def is_possible_closure(message: str) -> bool:
    """
    Identifica possíveis mensagens de encerramento para diagnóstico.
    """

    normalized = normalize_message(message)

    if not normalized:
        return False

    return any(
        term.lower() in normalized
        for term in CLOSURE_TERMS
    )


def is_protocol_event(message: str) -> bool:
    """
    Identifica mensagens técnicas relacionadas ao protocolo.
    """

    normalized = normalize_message(message)

    if not normalized:
        return False

    return (
        "protocolo de atendimento:" in normalized
        or "o tre-pb agradece" in normalized
    )


def is_evaluation_event(message: str) -> bool:
    """
    Identifica mensagens que solicitam avaliação do atendimento.
    """

    normalized = normalize_message(message)

    if not normalized:
        return False

    return (
        "avalie nosso atendimento" in normalized
        or "avalie o nosso atendimento" in normalized
        or "nota de 1 a 5" in normalized
    )


def is_agent_acceptance_message(
    message: str,
    origin: str,
) -> bool:
    """
    Identifica a mensagem disparada quando um atendente
    aceita a conversa.

    A mensagem é tecnicamente registrada como
    Atendimento automático, mas representa o aceite
    do atendimento humano.
    """

    if origin.strip() != "Atendimento automático":
        return False

    normalized = normalize_message(message)

    if not normalized:
        return False

    return all(
        term in normalized
        for term in AGENT_ACCEPTANCE_TERMS
    )


# ============================================================
# CLASSIFICAÇÃO DAS MENSAGENS
# ============================================================

def classify_message_type(row: pd.Series) -> str:
    """
    Classifica uma mensagem segundo sua natureza estrutural.
    """

    message = str(
        row.get("Mensagem", "")
    ).strip()

    origin = str(
        row.get("Origem", "")
    ).strip()

    if is_transfer_message(message):
        return "evento_transferencia"

    if is_inactivity_closure(message):
        return "evento_inatividade"

    if is_equivocal_closure(message):
        return "evento_encerramento_equivoco"

    if is_explicit_closure(message):
        return "evento_encerramento"

    if is_agent_acceptance_message(
        message,
        origin,
    ):
        return "evento_aceite_atendimento_humano"

    if is_protocol_event(message):
        return "evento_protocolo"

    if is_evaluation_event(message):
        return "evento_avaliacao"

    if is_audio_message(message):
        return "audio"

    if origin == "Contato":
        return "mensagem_contato"

    if origin == "Agente":
        return "mensagem_agente"

    if origin == "Atendimento automático":
        return "mensagem_automatica"

    if origin == "Não identificado":
        return "mensagem_nao_identificada"

    return "outro"


# ============================================================
# PREPARAÇÃO DAS MENSAGENS
# ============================================================

def prepare_messages(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Adiciona classificações estruturais às mensagens.
    """

    result = df.copy()

    result["Tipo de mensagem"] = result.apply(
        classify_message_type,
        axis=1,
    )

    return result


# ============================================================
# EXTRAÇÃO DO ATENDENTE
# ============================================================

def extract_agent_from_acceptance(
    message: str,
) -> str:
    """
    Tenta identificar o nome do atendente na mensagem de aceite.
    """

    text = str(message).strip()

    if not text:
        return ""

    normalized = text.lower()

    marker = "me chamo"

    if marker not in normalized:
        return ""

    start = normalized.find(marker)

    name_start = start + len(marker)

    remaining = text[name_start:].strip()

    if not remaining:
        return ""

    separators = [
        " e vou realizar",
        " e vou ",
        " vou realizar",
        ",",
        ".",
    ]

    end_positions = []

    remaining_lower = remaining.lower()

    for separator in separators:

        position = remaining_lower.find(
            separator
        )

        if position >= 0:
            end_positions.append(position)

    if end_positions:

        end = min(end_positions)

        name = remaining[:end].strip()

    else:

        name = remaining.strip()

    return name.strip(
        " ,.;:!?-"
    )


# ============================================================
# INFORMAÇÕES DO ATENDIMENTO HUMANO
# ============================================================

def get_human_interaction_data(
    group: pd.DataFrame,
) -> dict:
    """
    Identifica o início da interação humana através do
    primeiro evento de aceite.
    """

    acceptance_messages = group[
        group["Tipo de mensagem"]
        == "evento_aceite_atendimento_humano"
    ].copy()

    if acceptance_messages.empty:

        return {
            "Houve atendimento humano": False,
            "Quantidade aceites atendimento humano": 0,
            "Início atendimento humano": pd.NaT,
            "Atendente no aceite": "",
        }

    acceptance_messages = acceptance_messages.sort_values(
        by="Data da mensagem",
        kind="stable",
    )

    first_acceptance = acceptance_messages.iloc[0]

    start_human = first_acceptance[
        "Data da mensagem"
    ]

    agent_name = extract_agent_from_acceptance(
        first_acceptance["Mensagem"]
    )

    return {
        "Houve atendimento humano": True,
        "Quantidade aceites atendimento humano": int(
            len(acceptance_messages)
        ),
        "Início atendimento humano": start_human,
        "Atendente no aceite": agent_name,
    }


# ============================================================
# CÁLCULO DO TEMPO DENTRO DO EXPEDIENTE
# ============================================================

def calculate_working_minutes(
    start: pd.Timestamp,
    end: pd.Timestamp,
    allow_extension: bool = False,
) -> float:
    """
    Calcula minutos dentro do expediente entre dois timestamps.

    Expediente normal: 06:30 às 19:00.
    Extensão observacional: 19:00 às 20:05, quando permitida.
    Todos os dias da semana são considerados.
    """

    if (
        pd.isna(start)
        or pd.isna(end)
        or end <= start
    ):
        return 0.0

    total_minutes = 0.0
    current_date = start.normalize()
    final_date = end.normalize()

    while current_date <= final_date:
        normal_start = current_date + pd.Timedelta(
            hours=WORK_START_HOUR,
            minutes=WORK_START_MINUTE,
        )
        normal_end = current_date + pd.Timedelta(
            hours=WORK_END_HOUR,
            minutes=WORK_END_MINUTE,
        )
        extension_end = current_date + pd.Timedelta(
            hours=EXTENSION_END_HOUR,
            minutes=EXTENSION_END_MINUTE,
        )

        interval_start = max(start, normal_start)
        interval_end = min(end, normal_end)

        if interval_end > interval_start:
            total_minutes += (
                interval_end - interval_start
            ).total_seconds() / 60

        if allow_extension:
            interval_start = max(start, normal_end)
            interval_end = min(end, extension_end)

            if interval_end > interval_start:
                total_minutes += (
                    interval_end - interval_start
                ).total_seconds() / 60

        current_date += pd.Timedelta(days=1)

    return total_minutes


def calculate_off_hours_minutes(
    start: pd.Timestamp,
    end: pd.Timestamp,
    allow_extension: bool = False,
) -> float:
    """Calcula o tempo cronológico fora do expediente entre dois timestamps."""

    if (
        pd.isna(start)
        or pd.isna(end)
        or end <= start
    ):
        return 0.0

    total_minutes = (end - start).total_seconds() / 60
    working_minutes = calculate_working_minutes(
        start,
        end,
        allow_extension=allow_extension,
    )

    return max(0.0, total_minutes - working_minutes)


def _empty_gap_data() -> dict:
    """Retorna a estrutura padrão dos indicadores de intervalos."""
    return {
        "Quantidade intervalos interação": 0,
        "Maior intervalo entre mensagens": pd.NaT,
        "Maior intervalo minutos": 0.0,
        "Houve retomada outro dia": False,
        "Quantidade retomadas outro dia": 0,
        "Houve intervalo atravessando expediente": False,
    }


def _is_human_interaction(row: pd.Series) -> bool:
    """Identifica uma interação efetiva entre contato e agente."""
    return row.get("Tipo de mensagem", "") in {
        "mensagem_contato",
        "mensagem_agente",
        "audio",
    }


def analyze_interaction_gaps(
    group: pd.DataFrame,
    human_start: pd.Timestamp,
) -> dict:
    """
    Analisa intervalos entre interações efetivas.

    Intervalos intradia não são classificados como pausas.
    A única retomada especial é aquela que atravessa uma data
    e possui nova interação humana no dia seguinte.
    """

    result = _empty_gap_data()

    if pd.isna(human_start):
        return result

    interaction = group[
        group["Data da mensagem"] >= human_start
    ].copy()
    interaction = interaction[
        interaction.apply(_is_human_interaction, axis=1)
    ].sort_values(
        by="Data da mensagem",
        kind="stable",
    )

    if interaction.empty:
        return result

    timestamps = interaction["Data da mensagem"].dropna().tolist()
    if not timestamps:
        return result

    # O aceite representa o início do atendimento, mesmo sendo
    # tecnicamente uma mensagem automática.
    timestamps = [human_start] + [
        ts for ts in timestamps if ts != human_start
    ]

    gaps = []
    for index in range(1, len(timestamps)):
        previous = timestamps[index - 1]
        current = timestamps[index]
        gaps.append((previous, current, current - previous))

    if not gaps:
        return result

    longest_previous, longest_current, longest_gap = max(
        gaps,
        key=lambda item: item[2],
    )

    resumed_next_day = 0
    crosses_office_boundary = False

    for previous, current, _gap in gaps:
        if previous.date() != current.date():
            resumed_next_day += 1
            crosses_office_boundary = True

        if (
            previous.time() >= pd.Timestamp("19:00").time()
            or current.time() < pd.Timestamp("06:30").time()
        ):
            crosses_office_boundary = True

    result.update({
        "Quantidade intervalos interação": len(gaps),
        "Maior intervalo entre mensagens": (
            longest_current - longest_previous
        ),
        "Maior intervalo minutos": (
            longest_gap.total_seconds() / 60
        ),
        "Houve retomada outro dia": resumed_next_day > 0,
        "Quantidade retomadas outro dia": resumed_next_day,
        "Houve intervalo atravessando expediente": crosses_office_boundary,
    })

    return result


def calculate_effective_human_minutes(
    group: pd.DataFrame,
    human_start: pd.Timestamp,
) -> dict:
    """
    Calcula o tempo efetivo de interação humana.

    O início efetivo é a primeira mensagem do agente após o aceite.
    Interações dentro do mesmo dia permanecem no mesmo bloco.
    Quando há interação humana no dia seguinte, o intervalo noturno
    não é contabilizado: o bloco anterior termina às 19:00 e o novo
    bloco começa na primeira interação humana do dia seguinte.
    """

    result = {
        "Tempo efetivo atendimento humano": pd.NaT,
        "Tempo retomada entre dias": pd.Timedelta(0),
        "Quantidade blocos atendimento humano": 0,
        "Houve retomada outro dia": False,
    }

    if pd.isna(human_start):
        return result

    interaction = group[
        group["Data da mensagem"] >= human_start
    ].copy()

    interaction = interaction[
        interaction["Tipo de mensagem"].isin([
            "mensagem_contato",
            "mensagem_agente",
            "audio",
        ])
    ].sort_values(
        by="Data da mensagem",
        kind="stable",
    )

    if interaction.empty:
        return result

    # Para medir interação efetiva, o ponto inicial é a primeira
    # mensagem efetivamente enviada pelo agente.
    agent_rows = interaction[
        interaction["Origem"] == "Agente"
    ]

    if agent_rows.empty:
        return result

    effective_start = agent_rows["Data da mensagem"].min()
    timestamps = interaction[
        interaction["Data da mensagem"] >= effective_start
    ]["Data da mensagem"].dropna().tolist()

    if not timestamps:
        return result

    effective_seconds = 0.0
    resumption_seconds = 0.0
    blocks = 1

    for previous, current in zip(timestamps[:-1], timestamps[1:]):
        if current.date() == previous.date():
            effective_seconds += (current - previous).total_seconds()
            continue

        blocks += 1

        normal_day_end = previous.normalize() + pd.Timedelta(
            hours=WORK_END_HOUR,
            minutes=WORK_END_MINUTE,
        )

        extension_day_end = previous.normalize() + pd.Timedelta(
            hours=EXTENSION_END_HOUR,
            minutes=EXTENSION_END_MINUTE,
        )

        same_day_rows = interaction[
            interaction["Data da mensagem"].dt.date
            == previous.date()
        ]

        same_day_last = (
            same_day_rows["Data da mensagem"].max()
            if not same_day_rows.empty
            else previous
        )

        # A extensão somente existe se houve interação efetiva
        # naquele dia após as 19h.
        if same_day_last > normal_day_end:
            day_end = min(
                same_day_last,
                extension_day_end,
            )
        else:
            day_end = normal_day_end

        if previous < day_end:
            effective_seconds += (
                day_end - previous
            ).total_seconds()
            overnight_start = day_end
        else:
            overnight_start = previous

        resumption_seconds += max(
            0.0,
            (current - overnight_start).total_seconds(),
        )

    result.update({
        "Tempo efetivo atendimento humano": pd.Timedelta(
            seconds=effective_seconds
        ),
        "Tempo retomada entre dias": pd.Timedelta(
            seconds=resumption_seconds
        ),
        "Quantidade blocos atendimento humano": blocks,
        "Houve retomada outro dia": blocks > 1,
    })

    return result


def summarize_protocol(
    protocol: str,
    group: pd.DataFrame,
) -> dict:
    """
    Consolida todas as informações estruturais de um protocolo.
    """

    group = group.sort_values(
        by="Data da mensagem",
        kind="stable",
    ).copy()

    first_message = group[
        "Data da mensagem"
    ].min()

    last_message = group[
        "Data da mensagem"
    ].max()

    # --------------------------------------------------------
    # Duração bruta do protocolo
    # --------------------------------------------------------

    raw_duration = pd.NaT

    if (
        pd.notna(first_message)
        and pd.notna(last_message)
    ):

        raw_duration = (
            last_message
            - first_message
        )

    # --------------------------------------------------------
    # Quantidades por tipo
    # --------------------------------------------------------

    contact_messages = (
        group["Tipo de mensagem"]
        == "mensagem_contato"
    ).sum()

    agent_messages = (
        group["Tipo de mensagem"]
        == "mensagem_agente"
    ).sum()

    automatic_messages = (
        group["Tipo de mensagem"]
        == "mensagem_automatica"
    ).sum()

    unidentified_messages = (
        group["Tipo de mensagem"]
        == "mensagem_nao_identificada"
    ).sum()

    other_messages = (
        group["Tipo de mensagem"]
        == "outro"
    ).sum()

    audio_messages = (
        group["Tipo de mensagem"]
        == "audio"
    ).sum()

    transfer_events = (
        group["Tipo de mensagem"]
        == "evento_transferencia"
    ).sum()

    acceptance_events = (
        group["Tipo de mensagem"]
        == "evento_aceite_atendimento_humano"
    ).sum()

    explicit_closure_events = (
        group["Tipo de mensagem"]
        == "evento_encerramento"
    ).sum()

    inactivity_closure_events = (
        group["Tipo de mensagem"]
        == "evento_inatividade"
    ).sum()

    equivocal_closure_events = (
        group["Tipo de mensagem"]
        == "evento_encerramento_equivoco"
    ).sum()

    protocol_events = (
        group["Tipo de mensagem"]
        == "evento_protocolo"
    ).sum()

    evaluation_events = (
        group["Tipo de mensagem"]
        == "evento_avaliacao"
    ).sum()

    # --------------------------------------------------------
    # Atendimento humano
    # --------------------------------------------------------

    human_data = get_human_interaction_data(
        group
    )

    human_start = human_data[
        "Início atendimento humano"
    ]

    # --------------------------------------------------------
    # Encerramentos
    # --------------------------------------------------------

    closure_mask = (
        group["Tipo de mensagem"]
        .isin(CLOSURE_MESSAGE_TYPES)
    )

    closure_messages = group[
        closure_mask
    ].copy()

    first_closure = pd.NaT
    last_closure = pd.NaT
    closure_type = "NAO_IDENTIFICADO"

    if not closure_messages.empty:

        closure_messages = closure_messages.sort_values(
            by="Data da mensagem",
            kind="stable",
        )

        first_closure = closure_messages[
            "Data da mensagem"
        ].iloc[0]

        last_closure = closure_messages[
            "Data da mensagem"
        ].iloc[-1]

        last_closure_type = closure_messages[
            "Tipo de mensagem"
        ].iloc[-1]

        if (
            last_closure_type
            == "evento_inatividade"
        ):

            closure_type = "INATIVIDADE"

        elif (
            last_closure_type
            == "evento_encerramento_equivoco"
        ):

            closure_type = "EQUIVOCO"

        elif (
            last_closure_type
            == "evento_encerramento"
        ):

            closure_type = "AUTOMATICO"

    # --------------------------------------------------------
    # Duração baseada no encerramento detectado
    # --------------------------------------------------------

    human_duration = pd.NaT

    if (
        pd.notna(human_start)
        and pd.notna(last_closure)
        and last_closure >= human_start
    ):

        human_duration = (
            last_closure
            - human_start
        )

    # --------------------------------------------------------
    # Tempo do início humano até última mensagem
    # --------------------------------------------------------

    human_to_last_message = pd.NaT

    if (
        pd.notna(human_start)
        and pd.notna(last_message)
        and last_message >= human_start
    ):

        human_to_last_message = (
            last_message
            - human_start
        )

    # --------------------------------------------------------
    # Última interação efetiva registrada
    #
    # Aqui excluímos eventos puramente técnicos de encerramento,
    # protocolo e avaliação.
    #
    # A finalidade é encontrar o último momento em que houve
    # efetivamente uma mensagem/interação do fluxo.
    # --------------------------------------------------------

    interaction_types = [
        "mensagem_contato",
        "mensagem_agente",
        "audio",
    ]

    interaction_messages = group[
        (
            group["Tipo de mensagem"]
            .isin(interaction_types)
        )
        & (
            group["Data da mensagem"]
            >= human_start
        )
    ].copy()

    last_interaction = pd.NaT

    if not interaction_messages.empty:

        last_interaction = interaction_messages[
            "Data da mensagem"
        ].max()

    # --------------------------------------------------------
    # Tempo até última interação
    # --------------------------------------------------------

    human_to_last_interaction = pd.NaT

    if (
        pd.notna(human_start)
        and pd.notna(last_interaction)
        and last_interaction >= human_start
    ):

        human_to_last_interaction = (
            last_interaction
            - human_start
        )

    # --------------------------------------------------------
    # Tempo de expediente
    #
    # Os indicadores abaixo usam o intervalo cronológico entre o
    # início do atendimento humano e a última interação. Eles são
    # mantidos separados do tempo efetivo, pois a retomada noturna
    # é tratada especificamente acima.
    # --------------------------------------------------------

    working_minutes = 0.0
    off_hours_minutes = 0.0
    extension_minutes = 0.0

    if (
        pd.notna(human_start)
        and pd.notna(last_interaction)
    ):
        working_minutes = calculate_working_minutes(
            human_start,
            last_interaction,
            allow_extension=False,
        )

        effective_rows = group[
            (group["Data da mensagem"] >= human_start)
            & group["Tipo de mensagem"].isin([
                "mensagem_contato",
                "mensagem_agente",
                "audio",
            ])
        ].copy()

        # A extensão é calculada uma única vez por dia, usando
        # a última interação humana observada naquele dia.
        if not effective_rows.empty:
            effective_rows["Dia"] = (
                effective_rows["Data da mensagem"].dt.normalize()
            )

            for day, day_rows in effective_rows.groupby("Dia"):
                day_last = day_rows["Data da mensagem"].max()

                extension_start = (
                    day + pd.Timedelta(
                        hours=WORK_END_HOUR,
                        minutes=WORK_END_MINUTE,
                    )
                )

                extension_end = (
                    day + pd.Timedelta(
                        hours=EXTENSION_END_HOUR,
                        minutes=EXTENSION_END_MINUTE,
                    )
                )

                if day_last > extension_start:
                    extension_minutes += (
                        min(day_last, extension_end)
                        - extension_start
                    ).total_seconds() / 60

        total_elapsed_minutes = (
            last_interaction - human_start
        ).total_seconds() / 60

        off_hours_minutes = max(
            0.0,
            total_elapsed_minutes
            - working_minutes
            - extension_minutes,
        )

    # --------------------------------------------------------
    # Análise dos intervalos
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Análise das pausas
    # --------------------------------------------------------

    gap_data = analyze_interaction_gaps(
        group,
        human_start,
    )

    effective_human_data = calculate_effective_human_minutes(
        group,
        human_start,
    )

    # --------------------------------------------------------
    # Mensagens posteriores ao primeiro encerramento
    # --------------------------------------------------------

    messages_after_closure = 0
    contact_after_closure = 0
    agent_after_closure = 0

    if pd.notna(first_closure):

        after_closure = group[
            group["Data da mensagem"]
            > first_closure
        ]

        messages_after_closure = len(
            after_closure
        )

        contact_after_closure = (
            after_closure["Origem"]
            == "Contato"
        ).sum()

        agent_after_closure = (
            after_closure["Origem"]
            == "Agente"
        ).sum()

    # --------------------------------------------------------
    # Agentes
    # --------------------------------------------------------

    agents = (
        group.loc[
            group["Origem"] == "Agente",
            "Agente",
        ]
        .replace("", pd.NA)
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    # --------------------------------------------------------
    # Identificador
    # --------------------------------------------------------

    identifiers = (
        group["Identificador"]
        .replace("", pd.NA)
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    identifier = (
        identifiers[0]
        if identifiers
        else ""
    )

    # --------------------------------------------------------
    # Status do atendimento humano
    # --------------------------------------------------------

    if human_data["Houve atendimento humano"]:

        human_status = (
            "ATENDIMENTO_HUMANO_CONFIRMADO"
        )

    elif transfer_events > 0:

        human_status = (
            "TRANSFERIDO_SEM_ACEITE_DETECTADO"
        )

    else:

        human_status = (
            "SEM_TRANSFERENCIA_PARA_HUMANO"
        )

    # --------------------------------------------------------
    # Resposta humana antes do encerramento
    # --------------------------------------------------------

    agent_before_closure = False

    if pd.notna(first_closure):

        agent_before_closure = (
            (
                group["Origem"]
                == "Agente"
            )
            & (
                group["Data da mensagem"]
                <= first_closure
            )
        ).any()

    else:

        agent_before_closure = (
            agent_messages > 0
        )

    # --------------------------------------------------------
    # Atendimento humano sem Origem=Agente
    # --------------------------------------------------------

    human_without_agent_origin = (
        human_data["Houve atendimento humano"]
        and agent_messages == 0
    )

    return {
        "Protocolo": protocol,
        "Identificador": identifier,

        "Data início protocolo": first_message,
        "Data fim protocolo": last_message,

        "Duração bruta protocolo": raw_duration,

        "Quantidade mensagens": len(group),

        "Mensagens contato": int(
            contact_messages
        ),

        "Mensagens agente": int(
            agent_messages
        ),

        "Mensagens automáticas": int(
            automatic_messages
        ),

        "Mensagens não identificadas": int(
            unidentified_messages
        ),

        "Mensagens outras": int(
            other_messages
        ),

        "Quantidade áudios": int(
            audio_messages
        ),

        # ----------------------------------------------------
        # Atendimento humano
        # ----------------------------------------------------

        "Houve atendimento humano": (
            human_data[
                "Houve atendimento humano"
            ]
        ),

        "Status atendimento humano": (
            human_status
        ),

        "Quantidade aceites atendimento humano": (
            human_data[
                "Quantidade aceites atendimento humano"
            ]
        ),

        "Início atendimento humano": (
            human_start
        ),

        "Atendente no aceite": (
            human_data[
                "Atendente no aceite"
            ]
        ),

        "Atendimento humano sem mensagem Origem=Agente": (
            human_without_agent_origin
        ),

        # ----------------------------------------------------
        # Agentes
        # ----------------------------------------------------

        "Quantidade agentes": len(agents),

        "Agentes": " | ".join(agents),

        "Houve mensagem Origem=Agente": (
            agent_messages > 0
        ),

        # ----------------------------------------------------
        # Transferência
        # ----------------------------------------------------

        "Houve transferência": (
            transfer_events > 0
        ),

        "Quantidade transferências": int(
            transfer_events
        ),

        # ----------------------------------------------------
        # Encerramento
        # ----------------------------------------------------

        "Encerramento detectado": (
            not closure_messages.empty
        ),

        "Tipo encerramento detectado": (
            closure_type
        ),

        "Primeiro encerramento detectado": (
            first_closure
        ),

        "Último encerramento detectado": (
            last_closure
        ),

        "Encerramento explícito": (
            explicit_closure_events > 0
        ),

        "Encerramento por inatividade": (
            inactivity_closure_events > 0
        ),

        "Encerramento por equívoco": (
            equivocal_closure_events > 0
        ),

        "Quantidade encerramentos explícitos": int(
            explicit_closure_events
        ),

        "Quantidade encerramentos por inatividade": int(
            inactivity_closure_events
        ),

        "Quantidade encerramentos por equívoco": int(
            equivocal_closure_events
        ),

        # ----------------------------------------------------
        # Pós-encerramento
        # ----------------------------------------------------

        "Mensagens após primeiro encerramento": int(
            messages_after_closure
        ),

        "Mensagens contato após encerramento": int(
            contact_after_closure
        ),

        "Mensagens agente após encerramento": int(
            agent_after_closure
        ),

        "Houve interação após encerramento": (
            contact_after_closure > 0
            or agent_after_closure > 0
        ),

        # ----------------------------------------------------
        # Resposta humana
        # ----------------------------------------------------

        "Houve mensagem de agente antes do encerramento": (
            bool(agent_before_closure)
        ),

        # ----------------------------------------------------
        # Tempos de atendimento
        # ----------------------------------------------------

        "Duração atendimento humano detectado": (
            human_duration
        ),

        "Tempo do início humano até última mensagem": (
            human_to_last_message
        ),

        "Última interação": (
            last_interaction
        ),

        "Tempo do início humano até última interação": (
            human_to_last_interaction
        ),

        # ----------------------------------------------------
        # Expediente
        # ----------------------------------------------------

        "Tempo dentro do expediente": (
            pd.Timedelta(
                minutes=working_minutes
            )
        ),

        "Tempo fora do expediente": (
            pd.Timedelta(
                minutes=off_hours_minutes
            )
        ),

        "Tempo extensão 19h-20h05": (
            pd.Timedelta(
                minutes=extension_minutes
            )
        ),

        # ----------------------------------------------------
        # Intervalos e retomadas
        # ----------------------------------------------------

        "Quantidade intervalos interação": (
            gap_data["Quantidade intervalos interação"]
        ),

        "Maior intervalo entre mensagens": (
            gap_data["Maior intervalo entre mensagens"]
        ),

        "Maior intervalo minutos": (
            gap_data["Maior intervalo minutos"]
        ),

        "Houve retomada outro dia": (
            gap_data["Houve retomada outro dia"]
        ),

        "Quantidade retomadas outro dia": (
            gap_data["Quantidade retomadas outro dia"]
        ),

        "Houve intervalo atravessando expediente": (
            gap_data["Houve intervalo atravessando expediente"]
        ),

        "Tempo efetivo atendimento humano": (
            effective_human_data[
                "Tempo efetivo atendimento humano"
            ]
        ),

        "Tempo retomada entre dias": (
            effective_human_data[
                "Tempo retomada entre dias"
            ]
        ),

        "Quantidade blocos atendimento humano": (
            effective_human_data[
                "Quantidade blocos atendimento humano"
            ]
        ),

        # ----------------------------------------------------
        # Eventos técnicos
        # ----------------------------------------------------

        "Quantidade eventos protocolo": int(
            protocol_events
        ),

        "Quantidade eventos avaliação": int(
            evaluation_events
        ),
    }


# ============================================================
# CONSOLIDAÇÃO DOS PROTOCOLOS
# ============================================================

def build_protocol_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Cria um DataFrame com uma linha por protocolo.

    O agrupamento é exclusivamente por Protocolo.
    """

    if "Tipo de mensagem" in df.columns:

        prepared = df.copy()

    else:

        prepared = prepare_messages(df)

    protocol_rows = []

    for protocol, group in prepared.groupby(
        "Protocolo",
        sort=False,
    ):

        protocol_rows.append(
            summarize_protocol(
                protocol=str(protocol),
                group=group,
            )
        )

    summary = pd.DataFrame(
        protocol_rows
    )

    if summary.empty:
        return summary

    summary = summary.sort_values(
        by="Data início protocolo",
        kind="stable",
    ).reset_index(drop=True)

    return summary


# ============================================================
# DIAGNÓSTICO DE ENCERRAMENTOS
# ============================================================

def show_closure_diagnostics(
    df: pd.DataFrame,
) -> None:
    """
    Mostra diagnóstico quantitativo dos encerramentos.
    """

    print("\n" + "=" * 70)
    print("DIAGNÓSTICO DE ENCERRAMENTOS")
    print("=" * 70)

    explicit = df[
        df["Tipo de mensagem"]
        == "evento_encerramento"
    ]

    inactivity = df[
        df["Tipo de mensagem"]
        == "evento_inatividade"
    ]

    equivocal = df[
        df["Tipo de mensagem"]
        == "evento_encerramento_equivoco"
    ]

    possible = df[
        df["Mensagem"]
        .fillna("")
        .astype(str)
        .str.lower()
        .apply(
            lambda message: any(
                term.lower() in message
                for term in CLOSURE_TERMS
            )
        )
    ]

    acceptance = df[
        df["Tipo de mensagem"]
        == "evento_aceite_atendimento_humano"
    ]

    print(
        "Mensagens com termos de encerramento: "
        f"{len(possible):,}"
    )

    print(
        "Encerramentos normais detectados: "
        f"{len(explicit):,}"
    )

    print(
        "Encerramentos por inatividade detectados: "
        f"{len(inactivity):,}"
    )

    print(
        "Encerramentos por equívoco detectados: "
        f"{len(equivocal):,}"
    )

    print("\n" + "-" * 70)
    print("ACEITE DO ATENDIMENTO HUMANO")
    print("-" * 70)

    print(
        "Mensagens de aceite detectadas: "
        f"{len(acceptance):,}"
    )


# ============================================================
# DIAGNÓSTICO DA INTERAÇÃO HUMANA
# ============================================================

def show_human_interaction_diagnostics(
    summary: pd.DataFrame,
) -> None:
    """
    Mostra a distribuição dos protocolos em relação
    ao atendimento humano.
    """

    print("\n" + "=" * 70)
    print("DIAGNÓSTICO DA INTERAÇÃO HUMANA")
    print("=" * 70)

    confirmed = summary[
        summary["Houve atendimento humano"]
    ]

    transferred_without_acceptance = summary[
        summary["Status atendimento humano"]
        == "TRANSFERIDO_SEM_ACEITE_DETECTADO"
    ]

    no_transfer = summary[
        summary["Status atendimento humano"]
        == "SEM_TRANSFERENCIA_PARA_HUMANO"
    ]

    print(
        "Atendimento humano confirmado: "
        f"{len(confirmed):,}"
    )

    print(
        "Transferidos sem aceite detectado: "
        f"{len(transferred_without_acceptance):,}"
    )

    print(
        "Sem transferência para humano: "
        f"{len(no_transfer):,}"
    )

    print("\n" + "-" * 70)
    print("ATENDIMENTO HUMANO SEM ORIGEM=AGENTE")
    print("-" * 70)

    human_without_origin = summary[
        summary[
            "Atendimento humano sem mensagem Origem=Agente"
        ]
    ]

    print(
        "Protocolos nessa situação: "
        f"{len(human_without_origin):,}"
    )


# ============================================================
# DIAGNÓSTICO DO EXPEDIENTE E PAUSAS
# ============================================================

def show_schedule_diagnostics(
    summary: pd.DataFrame,
) -> None:
    """Mostra os principais indicadores relacionados ao expediente."""

    print("\n" + "=" * 70)
    print("DIAGNÓSTICO DE EXPEDIENTE")
    print("=" * 70)

    human = summary[summary["Houve atendimento humano"]].copy()

    print(
        "Protocolos com atendimento humano: "
        f"{len(human):,}"
    )

    resumed_next_day = human[human["Houve retomada outro dia"]]
    crossed_schedule = human[
        human["Houve intervalo atravessando expediente"]
    ]

    print(
        "Protocolos retomados em outro dia: "
        f"{len(resumed_next_day):,}"
    )

    print(
        "Protocolos com interrupção pelo fora do expediente: "
        f"{len(crossed_schedule):,}"
    )

    effective = (
        human["Tempo efetivo atendimento humano"]
        .dropna()
        .dt.total_seconds() / 60
    )

    if not effective.empty:
        print("\nTEMPO EFETIVO DE ATENDIMENTO HUMANO")
        print(f"  Média: {effective.mean():.2f} minutos")
        print(f"  Mediana: {effective.median():.2f} minutos")

    working = (
        human["Tempo dentro do expediente"]
        .dropna()
        .dt.total_seconds() / 60
    )

    off_hours = (
        human["Tempo fora do expediente"]
        .dropna()
        .dt.total_seconds() / 60
    )

    extension = (
        human["Tempo extensão 19h-20h05"]
        .dropna()
        .dt.total_seconds() / 60
    )

    if not working.empty:
        print("\nTEMPO NO EXPEDIENTE NORMAL (06h30-19h)")
        print(f"  Média: {working.mean():.2f} minutos")
        print(f"  Mediana: {working.median():.2f} minutos")

    if not extension.empty:
        print("\nTEMPO DE EXTENSÃO OBSERVADO (19h-20h05)")
        print(f"  Média: {extension.mean():.2f} minutos")
        print(f"  Mediana: {extension.median():.2f} minutos")

    if not off_hours.empty:
        print("\nTEMPO FORA DO EXPEDIENTE (19h-06h30)")
        print(f"  Média: {off_hours.mean():.2f} minutos")
        print(f"  Mediana: {off_hours.median():.2f} minutos")


def show_post_closure_diagnostics(
    summary: pd.DataFrame,
) -> None:
    """
    Mostra protocolos com mensagens após o primeiro
    encerramento detectado.
    """

    with_post_interaction = summary[
        summary[
            "Houve interação após encerramento"
        ]
    ]

    with_post_messages = summary[
        summary[
            "Mensagens após primeiro encerramento"
        ] > 0
    ]

    print("\n" + "=" * 70)
    print("DIAGNÓSTICO PÓS-ENCERRAMENTO")
    print("=" * 70)

    print(
        "Protocolos com qualquer mensagem após o "
        "primeiro encerramento: "
        f"{len(with_post_messages):,}"
    )

    print(
        "Protocolos com interação de contato/agente "
        "após o encerramento: "
        f"{len(with_post_interaction):,}"
    )

    if not with_post_interaction.empty:

        print("\nDistribuição por tipo de encerramento:")

        print(
            with_post_interaction[
                "Tipo encerramento detectado"
            ]
            .value_counts()
            .to_string()
        )


# ============================================================
# RELATÓRIO GERAL
# ============================================================

def show_protocol_report(
    df: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    """
    Exibe o diagnóstico geral dos protocolos.
    """

    print("\n" + "=" * 70)
    print("DIAGNÓSTICO DOS PROTOCOLOS")
    print("=" * 70)

    print(
        f"Mensagens analisadas: {len(df):,}"
    )

    print(
        f"Protocolos analisados: {len(summary):,}"
    )

    # --------------------------------------------------------
    # Origem
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("ORIGEM DAS MENSAGENS")
    print("-" * 70)

    print(
        df["Origem"]
        .replace("", "[VAZIO]")
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # Tipo técnico
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("TIPO TÉCNICO DAS MENSAGENS")
    print("-" * 70)

    print(
        df["Tipo de mensagem"]
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # Atendimento humano
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("ATENDIMENTO HUMANO")
    print("-" * 70)

    print(
        summary[
            "Status atendimento humano"
        ]
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # Mensagens de agente
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("MENSAGENS DE AGENTE")
    print("-" * 70)

    with_agent_origin = summary[
        summary[
            "Houve mensagem Origem=Agente"
        ]
    ]

    without_agent_origin = summary[
        ~summary[
            "Houve mensagem Origem=Agente"
        ]
    ]

    print(
        "Protocolos com Origem=Agente: "
        f"{len(with_agent_origin):,}"
    )

    print(
        "Protocolos sem Origem=Agente: "
        f"{len(without_agent_origin):,}"
    )

    # --------------------------------------------------------
    # Aceite
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("ACEITE DO ATENDIMENTO")
    print("-" * 70)

    accepted = summary[
        summary[
            "Houve atendimento humano"
        ]
    ]

    print(
        "Protocolos com aceite detectado: "
        f"{len(accepted):,}"
    )

    # --------------------------------------------------------
    # Transferências
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("TRANSFERÊNCIAS")
    print("-" * 70)

    transferred = summary[
        summary["Houve transferência"]
    ]

    total_transfers = int(
        summary[
            "Quantidade transferências"
        ].sum()
    )

    print(
        "Protocolos com transferência: "
        f"{len(transferred):,}"
    )

    print(
        "Eventos de transferência: "
        f"{total_transfers:,}"
    )

    # --------------------------------------------------------
    # Áudios
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("ÁUDIOS")
    print("-" * 70)

    protocols_with_audio = summary[
        summary["Quantidade áudios"] > 0
    ]

    total_audio = int(
        summary[
            "Quantidade áudios"
        ].sum()
    )

    print(
        "Mensagens de áudio: "
        f"{total_audio:,}"
    )

    print(
        "Protocolos com áudio: "
        f"{len(protocols_with_audio):,}"
    )

    # --------------------------------------------------------
    # Encerramentos
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("ENCERRAMENTOS")
    print("-" * 70)

    closure_protocols = summary[
        summary["Encerramento detectado"]
    ]

    print(
        "Protocolos com encerramento detectado: "
        f"{len(closure_protocols):,}"
    )

    print("\nTipos de encerramento:")

    print(
        summary[
            "Tipo encerramento detectado"
        ]
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # Pós-encerramento
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("PÓS-ENCERRAMENTO")
    print("-" * 70)

    post_closure = summary[
        summary[
            "Houve interação após encerramento"
        ]
    ]

    print(
        "Protocolos com interação após encerramento: "
        f"{len(post_closure):,}"
    )

    # --------------------------------------------------------
    # Tempos
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("TEMPOS")
    print("-" * 70)

    raw_durations = (
        summary[
            "Duração bruta protocolo"
        ]
        .dropna()
        .dt.total_seconds()
        / 60
    )

    human_durations = (
        summary[
            "Duração atendimento humano detectado"
        ]
        .dropna()
        .dt.total_seconds()
        / 60
    )

    if not raw_durations.empty:

        print(
            "DURAÇÃO BRUTA DO PROTOCOLO"
        )

        print(
            f"  Média: "
            f"{raw_durations.mean():.2f} minutos"
        )

        print(
            f"  Mediana: "
            f"{raw_durations.median():.2f} minutos"
        )

        print(
            f"  Máxima: "
            f"{raw_durations.max():.2f} minutos"
        )

    if not human_durations.empty:

        print(
            "\nDURAÇÃO DO ATENDIMENTO HUMANO "
            "DETECTADO"
        )

        print(
            f"  Média: "
            f"{human_durations.mean():.2f} minutos"
        )

        print(
            f"  Mediana: "
            f"{human_durations.median():.2f} minutos"
        )

        print(
            f"  Mínima: "
            f"{human_durations.min():.2f} minutos"
        )

        print(
            f"  Máxima: "
            f"{human_durations.max():.2f} minutos"
        )

    # --------------------------------------------------------
    # Agentes
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("AGENTES")
    print("-" * 70)

    distinct_agents = (
        df.loc[
            df["Origem"] == "Agente",
            "Agente",
        ]
        .replace("", pd.NA)
        .dropna()
        .astype(str)
        .str.strip()
        .nunique()
    )

    print(
        "Agentes distintos no CSV: "
        f"{distinct_agents:,}"
    )

    protocols_multiple_agents = summary[
        summary["Quantidade agentes"] > 1
    ]

    print(
        "Protocolos com mais de um agente: "
        f"{len(protocols_multiple_agents):,}"
    )

    print(
        "Total de participações de agentes "
        "nos protocolos: "
        f"{int(summary['Quantidade agentes'].sum()):,}"
    )

    # --------------------------------------------------------
    # Identificadores
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("IDENTIFICADORES")
    print("-" * 70)

    protocols_per_identifier = (
        summary
        .groupby("Identificador")
        ["Protocolo"]
        .nunique()
    )

    repeated_identifiers = (
        protocols_per_identifier[
            protocols_per_identifier > 1
        ]
    )

    print(
        "Identificadores com mais de um protocolo: "
        f"{len(repeated_identifiers):,}"
    )

    if not repeated_identifiers.empty:

        print(
            "Maior quantidade de protocolos por "
            "identificador: "
            f"{int(repeated_identifiers.max()):,}"
        )

    print("\n" + "=" * 70)


# ============================================================
# EXECUÇÃO DIRETA
# ============================================================

def main() -> None:
    """
    Executa o diagnóstico estrutural dos protocolos.
    """

    print("=" * 70)
    print("SERVICE DESK ANALYZER")
    print("MAPEAMENTO DOS PROTOCOLOS")
    print("=" * 70)

    # --------------------------------------------------------
    # Carrega e prepara o CSV
    # --------------------------------------------------------

    df = prepare_conversations(
        DEFAULT_CSV_PATH
    )

    # --------------------------------------------------------
    # Classifica as mensagens
    # --------------------------------------------------------

    prepared = prepare_messages(
        df
    )

    # --------------------------------------------------------
    # Consolida os protocolos
    # --------------------------------------------------------

    summary = build_protocol_summary(
        prepared
    )

    # --------------------------------------------------------
    # Relatório geral
    # --------------------------------------------------------

    show_protocol_report(
        prepared,
        summary,
    )

    # --------------------------------------------------------
    # Diagnóstico de encerramentos
    # --------------------------------------------------------

    show_closure_diagnostics(
        prepared
    )

    # --------------------------------------------------------
    # Diagnóstico da interação humana
    # --------------------------------------------------------

    show_human_interaction_diagnostics(
        summary
    )

    # --------------------------------------------------------
    # Diagnóstico de expediente e pausas
    # --------------------------------------------------------

    show_schedule_diagnostics(
        summary
    )

    # --------------------------------------------------------
    # Diagnóstico pós-encerramento
    # --------------------------------------------------------

    show_post_closure_diagnostics(
        summary
    )

    print("\n" + "=" * 70)
    print("MAPEAMENTO CONCLUÍDO")
    print("=" * 70)


if __name__ == "__main__":
    main()

