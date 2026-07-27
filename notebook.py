"""
COLE TODO ESTE ARQUIVO EM UMA UNICA CELULA DO GOOGLE COLAB E EXECUTE-A.

Arquitetura hibrida:
1) Moonshot/Kimi (API) atua como professor e cria dados sinteticos a partir do
   texto-base sobre UP 4 SBI.
2) A GPU T4 do Colab ajusta, por QLoRA, um modelo-aluno aberto de 1,5B.

Isto e destilacao por respostas (SFT), nao treinamento nem copia dos pesos do
Kimi. A API da Moonshot nao fornece gradientes ou os pesos do modelo hospedado.

Documentacao consultada:
- https://platform.kimi.com/docs/api/overview
- https://platform.kimi.com/docs/models
- https://huggingface.co/docs/peft/developer_guides/quantization
"""

# ============================ CONFIGURACAO ============================
STUDENT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = "/content/up4sbi-qwen2.5-1.5b-lora"
N_SYNTHETIC_EXAMPLES = 48       # 6 chamadas de API, em condicoes normais
EXAMPLES_PER_API_CALL = 8
MAX_SEQUENCE_LENGTH = 1024
EPOCHS = 2
SEED = 42
MOONSHOT_BASE_URL = "https://api.moonshot.cn/v1"
# Deixe vazio para detectar pela API: prefere kimi-k3 quando disponivel e,
# caso contrario, usa o modelo oficial mais recente encontrado (ex.: kimi-k2.6).
MOONSHOT_MODEL_OVERRIDE = ""
# =====================================================================

import os
import sys
import json
import math
import time
import random
import hashlib
import getpass
import shutil
import subprocess
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)

print("Instalando dependencias (pode levar alguns minutos na primeira execucao)...")
subprocess.run(
    [
        sys.executable, "-m", "pip", "install", "-q", "-U",
        "transformers>=4.48,<5",
        "peft>=0.14,<1",
        "accelerate>=1.2,<2",
        "bitsandbytes>=0.45,<1",
        "datasets>=3,<5",
        "openai>=1.60,<3",
    ],
    check=True,
)

import torch
from datasets import Dataset
from openai import OpenAI
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)

set_seed(SEED)

if not torch.cuda.is_available():
    raise RuntimeError(
        "GPU nao encontrada. No Colab, selecione Ambiente de execucao > "
        "Alterar tipo de ambiente de execucao > GPU T4 e execute novamente."
    )

gpu_name = torch.cuda.get_device_name(0)
gpu_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
print(f"GPU: {gpu_name} ({gpu_gb:.1f} GiB)")
if "T4" not in gpu_name.upper():
    print("Aviso: o notebook foi dimensionado para T4; outra GPU tambem pode funcionar.")
if gpu_gb < 13:
    raise RuntimeError("A GPU tem pouca VRAM. Este roteiro requer aproximadamente 13 GiB ou mais.")

BASE_TXT = r"""
**Ajuste fino de modelos em uma arquitetura do tipo Unified Protocol for Synthetic Biological Intelligence (UP 4 SBI)**

### 1. Introdução
O ajuste fino é a adaptação de um modelo pré-treinado para uma tarefa, domínio ou estilo específico, sendo uma técnica central em IA generativa e em sistemas de aprendizado por transferência. Em SBI, a ideia de ajuste fino precisa ser vista de modo mais amplo: não apenas como adaptação de pesos em modelos digitais, mas como calibração de interfaces, protocolos de feedback, decodificação e controle em um sistema bio-digital fechado. Assim, uma arquitetura UP 4 SBI pode ser interpretada como uma camada unificadora para padronizar como modelos, sensores, atuadores e substratos biológicos interagem e aprendem.

### 2. Tendências centrais
A primeira tendência é a consolidação de ajuste fino eficiente em parâmetros. Métodos parciais e aditivos, como LoRA, adaptadores, prompts treináveis e quantização, reduzem custo computacional, memória e tempo de treinamento, mantendo bom desempenho em tarefas específicas. A segunda tendência é o crescimento de técnicas orientadas por preferência, como DPO, que alinham saídas a preferências humanas ou institucionais sem exigir, em todos os casos, um modelo de recompensa separado. A terceira tendência é o avanço do RFT, especialmente para domínios em que existe uma resposta correta objetiva e o raciocínio precisa ser otimizado por sinais de recompensa mais ricos.

### 3. Implicações para UP 4 SBI
Em uma arquitetura UP 4 SBI, o ajuste fino não deve ser tratado como um passo isolado, mas como parte de um ciclo de adaptação contínua entre codificação, inferência e retroalimentação. Isso é especialmente relevante em SBI porque o substrato biológico é dinâmico, não estacionário e sensível a ruído, o que exige protocolos de adaptação mais robustos do que os usados em modelos puramente digitais. Nesse contexto, modelos de IA podem ser afinados para interpretar sinais neurais, selecionar estímulos de feedback e ajustar políticas de interação com o tecido biológico. A tendência, portanto, é migrar de um ajuste fino estático para um ajuste fino sistêmico, acoplado à observação em tempo real.

### 4. Arquiteturas e modalidades
Outra tendência importante é a separação entre treinamento supervisionado, preferências e reforço, cada uma adequada a um tipo de problema. Para UP 4 SBI, isso sugere uma arquitetura modular: um bloco para interpretação de sinais, um bloco para decisão de intervenção e um bloco para avaliação de resultado biológico. Em modelos multimodais, também cresce o uso de ajuste fino com entradas de texto + imagem, o que é útil para documentação experimental, leitura de gráficos, inspeção de culturas e análise de dados visuais de laboratórios. Isso favorece aplicações híbridas em que o modelo precisa entender tanto linguagem técnica quanto evidência experimental.

### 5. Desafios técnicos
O principal desafio é a estabilidade do sistema: no SBI, mudanças no substrato podem alterar o próprio comportamento do sistema, exigindo reavaliação constante dos dados de treino e dos critérios de sucesso. Outro desafio é a reprodutibilidade, pois plataformas biológicas ainda dependem de condições experimentais muito controladas, com variação entre culturas, eletrodos e protocolos. Também há uma tensão entre eficiência e controle: métodos leves como LoRA facilitam iteração rápida, mas nem sempre capturam fenômenos complexos de alinhamento de longo prazo. Em termos de governança, a tendência é combinar ajustes finos técnicos com protocolos de segurança, rastreabilidade e conformidade, especialmente em ambientes biomédicos.

### 6. Linha de pesquisa proposta
Uma boa formulação de linha de pesquisa seria: "Desenvolvimento de protocolos de ajuste fino adaptativo para arquiteturas bio-digitais em laço fechado". Essa agenda permite investigar desde PEFT e DPO em modelos digitais até mecanismos de controle e avaliação em SBI, com foco em robustez, interpretabilidade e eficiência. Também abre espaço para comparar estratégias de SFT, preferência e reforço em cenários onde a entrada e a saída incluem sinais biológicos, texto técnico e feedback temporal. Em termos acadêmicos, é uma proposta forte porque articula teoria de modelos, engenharia de sistemas e experimentação interdisciplinar.

### 7. Conclusão
As tendências mais relevantes apontam para ajuste fino mais eficiente, modular, orientado por feedback e multimodal, com forte integração entre IA generativa e sistemas bio-digitais. Para uma arquitetura do tipo UP 4 SBI, o caminho mais promissor é tratar o ajuste fino como parte de um protocolo unificado de interação, e não apenas como uma etapa de treino de modelo. Isso fortalece tanto a aplicabilidade científica quanto a relevância institucional da proposta.
""".strip()

# Se o usuario enviar um base.txt para /content, ele substitui o texto incorporado.
uploaded_base = Path("/content/base.txt")
if uploaded_base.exists():
    BASE_TXT = uploaded_base.read_text(encoding="utf-8")
    print("Usando /content/base.txt enviado pelo usuario.")
else:
    print("Usando a versao de base.txt incorporada ao notebook.")

if len(BASE_TXT) < 500:
    raise ValueError("O texto-base e curto demais para gerar um conjunto de treino util.")

def get_moonshot_key():
    key = os.environ.get("MOONSHOT_API_KEY", "").strip()
    if not key:
        try:
            from google.colab import userdata
            key = (userdata.get("MOONSHOT_API_KEY") or "").strip()
        except Exception:
            pass
    if not key:
        key = getpass.getpass(
            "Cole sua MOONSHOT_API_KEY (ela nao sera exibida nem salva no dataset): "
        ).strip()
    if not key:
        raise RuntimeError("MOONSHOT_API_KEY nao informada.")
    return key

client = OpenAI(
    api_key=get_moonshot_key(),
    base_url=MOONSHOT_BASE_URL,
    timeout=180.0,
    max_retries=2,
)

def choose_teacher_model():
    if MOONSHOT_MODEL_OVERRIDE.strip():
        return MOONSHOT_MODEL_OVERRIDE.strip()
    available = sorted({m.id for m in client.models.list().data})
    print("Modelos Kimi visiveis na conta:", [m for m in available if "kimi" in m.lower()])
    exact_preferences = ["kimi-k3", "kimi-k2.6", "kimi-k2.5"]
    for wanted in exact_preferences:
        if wanted in available:
            return wanted
    k3_variants = [m for m in available if "kimi-k3" in m.lower()]
    if k3_variants:
        return k3_variants[-1]
    kimi_models = [m for m in available if "kimi" in m.lower()]
    if kimi_models:
        return kimi_models[-1]
    raise RuntimeError(
        "Nenhum modelo Kimi foi exposto por esta chave. Confira a regiao, o saldo "
        "e a chave da plataforma Moonshot."
    )

teacher_model = choose_teacher_model()
print(f"Professor selecionado: {teacher_model}")
print(
    f"A etapa remota enviara o texto-base a Moonshot e tentara produzir "
    f"{N_SYNTHETIC_EXAMPLES} exemplos. O uso da API pode gerar cobranca."
)

themes = [
    "definicoes fundamentais e explicacoes para estudantes",
    "comparacoes entre SFT, PEFT/LoRA, DPO e RFT",
    "arquitetura modular, sensores, atuadores e laco fechado",
    "estabilidade, nao estacionariedade, ruido e reprodutibilidade",
    "governanca, rastreabilidade, seguranca e limites das evidencias",
    "perguntas de pesquisa, hipoteses, metricas e desenho de avaliacao",
    "multimodalidade e interpretacao cautelosa de evidencia experimental",
    "critica academica: premissas, lacunas, riscos e melhorias do texto",
]

SYSTEM_TEACHER = """
Voce e um professor-pesquisador que prepara dados de alta qualidade para
ajuste supervisionado de um assistente academico em portugues brasileiro.
Seja rigoroso, claro, prudente e fiel apenas ao texto-fonte. Nao invente
resultados, referencias, dados biologicos, capacidades clinicas ou afirmacoes
de que a arquitetura ja foi validada. Diferencie proposta conceitual de
evidencia empirica. Nao revele raciocinio interno; entregue somente respostas
finais bem justificadas.
""".strip()

def extract_json_array(text):
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```json", "", 1).replace("```", "").strip()
    start, end = cleaned.find("["), cleaned.rfind("]")
    if start < 0 or end <= start:
        raise ValueError("A resposta nao contem um array JSON.")
    return json.loads(cleaned[start:end + 1])

def valid_example(item):
    if not isinstance(item, dict):
        return False
    user = item.get("user", "")
    assistant = item.get("assistant", "")
    if not isinstance(user, str) or not isinstance(assistant, str):
        return False
    if not (25 <= len(user.strip()) <= 700 and 120 <= len(assistant.strip()) <= 3500):
        return False
    forbidden = ["como uma ia", "nao posso ajudar", "```json"]
    return not any(x in assistant.lower() for x in forbidden)

def ask_teacher(batch_size, theme, batch_number):
    prompt = f"""
Com base EXCLUSIVAMENTE no TEXTO-FONTE abaixo, crie {batch_size} exemplos
diversos de instrucao e resposta para destilar seu conhecimento e seu estilo
academico em um modelo menor.

FOCO DESTE LOTE: {theme}

Requisitos:
- escreva em portugues brasileiro;
- varie entre explicacao, sintese, comparacao, critica, classificacao,
  formulacao de hipotese, proposta de metrica e analise de cenario;
- cada pergunta deve ser autocontida e util; evite perguntas quase duplicadas;
- respostas devem ter entre 1 e 5 paragrafos, ser tecnicamente prudentes e
  distinguir hipotese, proposta e evidencia;
- nao inclua citacoes ou links que nao estejam verificaveis no texto;
- nao forneca protocolos clinicos nem laboratoriais acionaveis;
- devolva SOMENTE um array JSON valido, sem markdown, exatamente no formato:
  [{{"user":"pergunta","assistant":"resposta"}}]

TEXTO-FONTE:
<<<
{BASE_TXT}
>>>

Identificador do lote: {batch_number}
""".strip()
    last_error = None
    for attempt in range(4):
        try:
            response = client.chat.completions.create(
                model=teacher_model,
                messages=[
                    {"role": "system", "content": SYSTEM_TEACHER},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.75,
                max_tokens=5000,
            )
            items = extract_json_array(response.choices[0].message.content)
            good = [x for x in items if valid_example(x)]
            if not good:
                raise ValueError("O lote nao trouxe exemplos validos.")
            return good
        except Exception as exc:
            last_error = exc
            wait_s = 2 ** attempt
            print(f"  tentativa {attempt + 1} falhou: {type(exc).__name__}; repetindo...")
            time.sleep(wait_s)
    raise RuntimeError(f"Falha persistente ao gerar lote: {last_error}")

examples = []
seen = set()
batch_number = 0
max_batches = math.ceil(N_SYNTHETIC_EXAMPLES / EXAMPLES_PER_API_CALL) + 4
while len(examples) < N_SYNTHETIC_EXAMPLES and batch_number < max_batches:
    theme = themes[batch_number % len(themes)]
    needed = N_SYNTHETIC_EXAMPLES - len(examples)
    requested = min(EXAMPLES_PER_API_CALL, needed)
    print(
        f"Gerando lote {batch_number + 1}/{max_batches}: "
        f"{requested} exemplos sobre {theme}..."
    )
    for item in ask_teacher(requested, theme, batch_number + 1):
        normalized_question = " ".join(item["user"].lower().split())
        fingerprint = hashlib.sha256(normalized_question.encode("utf-8")).hexdigest()
        if fingerprint not in seen:
            seen.add(fingerprint)
            examples.append(
                {
                    "user": item["user"].strip(),
                    "assistant": item["assistant"].strip(),
                    "teacher_model": teacher_model,
                    "source": "base.txt",
                }
            )
        if len(examples) >= N_SYNTHETIC_EXAMPLES:
            break
    batch_number += 1
    print(f"  total validado: {len(examples)}/{N_SYNTHETIC_EXAMPLES}")

if len(examples) < max(16, int(N_SYNTHETIC_EXAMPLES * 0.75)):
    raise RuntimeError(
        f"Foram obtidos apenas {len(examples)} exemplos; quantidade insuficiente "
        "para prosseguir de forma responsavel."
    )

random.shuffle(examples)
dataset_path = Path("/content/up4sbi_synthetic.jsonl")
with dataset_path.open("w", encoding="utf-8") as f:
    for row in examples:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"Dataset auditavel salvo em {dataset_path} ({len(examples)} exemplos).")

SYSTEM_STUDENT = """
Voce e um assistente academico especializado na proposta conceitual UP 4 SBI.
Responda em portugues brasileiro com clareza, rigor e prudencia. Diferencie
hipoteses, propostas e evidencias; nao invente validacao experimental,
referencias ou capacidades clinicas. Quando faltar evidencia, explicite a
limitacao. Priorize ajuste fino eficiente, modularidade, feedback em laco
fechado, estabilidade, reprodutibilidade, rastreabilidade e seguranca.
""".strip()

tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL, use_fast=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

def tokenize_example(row):
    prompt_messages = [
        {"role": "system", "content": SYSTEM_STUDENT},
        {"role": "user", "content": row["user"]},
    ]
    full_messages = prompt_messages + [
        {"role": "assistant", "content": row["assistant"]}
    ]
    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages, tokenize=True, add_generation_prompt=True
    )
    full_ids = tokenizer.apply_chat_template(
        full_messages, tokenize=True, add_generation_prompt=False
    )
    full_ids = full_ids[:MAX_SEQUENCE_LENGTH]
    # Mascara a instrucao: a perda e calculada somente na resposta do professor.
    common_prefix = 0
    for a, b in zip(prompt_ids, full_ids):
        if a != b:
            break
        common_prefix += 1
    labels = [-100] * common_prefix + full_ids[common_prefix:]
    if all(x == -100 for x in labels):
        raise ValueError("Exemplo truncado antes da resposta; aumente MAX_SEQUENCE_LENGTH.")
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }

dataset = Dataset.from_list(examples)
split = dataset.train_test_split(test_size=max(4, round(len(dataset) * 0.12)), seed=SEED)
train_ds = split["train"].map(tokenize_example, remove_columns=dataset.column_names)
eval_ds = split["test"].map(tokenize_example, remove_columns=dataset.column_names)

class CompletionOnlyCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        max_len = max(len(x["input_ids"]) for x in features)
        input_ids, attention_mask, labels = [], [], []
        for item in features:
            pad = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.tokenizer.pad_token_id] * pad)
            attention_mask.append(item["attention_mask"] + [0] * pad)
            labels.append(item["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

print(f"Carregando modelo-aluno {STUDENT_MODEL} em 4 bits...")
model = AutoModelForCausalLM.from_pretrained(
    STUDENT_MODEL,
    quantization_config=quant_config,
    device_map={"": 0},
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)
model = prepare_model_for_kbit_training(
    model, use_gradient_checkpointing=True
)
model.config.use_cache = False

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.08,
    weight_decay=0.01,
    max_grad_norm=0.3,
    fp16=True,
    bf16=False,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    logging_steps=1,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    report_to="none",
    seed=SEED,
    data_seed=SEED,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=CompletionOnlyCollator(tokenizer),
)

print("Medindo perda antes do ajuste...")
before = trainer.evaluate()
print(f"eval_loss antes: {before['eval_loss']:.4f}")

print("Iniciando QLoRA na T4...")
train_result = trainer.train()

print("Medindo perda depois do ajuste...")
after = trainer.evaluate()
print(f"eval_loss depois: {after['eval_loss']:.4f}")
if math.isfinite(after["eval_loss"]):
    print(f"perplexidade final aproximada: {math.exp(min(after['eval_loss'], 20)):.2f}")

final_adapter = Path(OUTPUT_DIR) / "final_adapter"
trainer.save_model(str(final_adapter))
tokenizer.save_pretrained(str(final_adapter))

metrics = {
    "teacher_model": teacher_model,
    "student_model": STUDENT_MODEL,
    "synthetic_examples": len(examples),
    "train_examples": len(train_ds),
    "eval_examples": len(eval_ds),
    "eval_loss_before": float(before["eval_loss"]),
    "eval_loss_after": float(after["eval_loss"]),
    "epochs": EPOCHS,
    "max_sequence_length": MAX_SEQUENCE_LENGTH,
    "method": "response distillation + supervised QLoRA",
    "limitation": (
        "O adaptador aprende o dominio e o estilo dos exemplos; ele nao replica "
        "a escala, a arquitetura, os pesos nem todas as capacidades do Kimi."
    ),
}
(final_adapter / "training_summary.json").write_text(
    json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
)

model.config.use_cache = True
model.eval()
test_question = (
    "Como o ajuste fino sistemico se diferencia do ajuste fino estatico "
    "em uma proposta UP 4 SBI?"
)
test_messages = [
    {"role": "system", "content": SYSTEM_STUDENT},
    {"role": "user", "content": test_question},
]
inputs = tokenizer.apply_chat_template(
    test_messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)
with torch.inference_mode():
    generated = model.generate(
        inputs,
        max_new_tokens=260,
        do_sample=False,
        repetition_penalty=1.08,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
answer = tokenizer.decode(generated[0, inputs.shape[-1]:], skip_special_tokens=True)
print("\n=== TESTE DO MODELO AJUSTADO ===")
print("Pergunta:", test_question)
print("Resposta:", answer)

archive = shutil.make_archive("/content/up4sbi_modelo_qlora", "zip", OUTPUT_DIR)
print("\nConcluido.")
print("Adaptador LoRA:", final_adapter)
print("Dataset sintetico:", dataset_path)
print("Arquivo para download:", archive)

try:
    from google.colab import files
    print("Iniciando download do ZIP do adaptador...")
    files.download(archive)
except Exception:
    print("Fora do Colab: baixe manualmente o ZIP indicado acima.")
