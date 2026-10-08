# Importação local CNES (experimental)

Fonte: [CNES, Dados Abertos do Ministério da Saúde](https://dadosabertos.saude.gov.br/dataset/cnes-cadastro-nacional-de-estabelecimentos-de-saude).

## Recurso conferido em 07/10/2026

O recurso oficial `cnes_estabelecimentos_csv.zip` contém um único
`cnes_estabelecimentos.csv` com `;` como separador e texto Latin-1. O cabeçalho
observado mapeia `CO_CNES`, `NO_FANTASIA`, `TP_UNIDADE`, `NO_LOGRADOURO`,
`NU_ENDERECO`, `NO_BAIRRO`, `CO_IBGE`, `CO_UF`, `NU_LATITUDE`, `NU_LONGITUDE`
e `CO_AMBULATORIAL_SUS`. A coluna `CO_MOTIVO_DESAB` também é exigida; linhas
com motivo de desabilitação preenchido são excluídas. `CO_IBGE` e `CO_UF` são códigos, exibidos como tal
no endereço. `CO_CNES` vem sem zeros à esquerda em várias linhas; o importador
normaliza para sete dígitos. A URL de download é
`https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/CNES/cnes_estabelecimentos_csv.zip`.

Use `--official-establishments` para esse esquema conferido. Um recurso futuro
com outro cabeçalho exige nova inspeção e mapeamento explícito.

Acesse a página oficial e baixe o CSV. **Inspecione o cabeçalho e o significado
de cada coluna antes de mapear**. Os recursos podem mudar de estrutura.
Não use uma coluna de natureza jurídica ou propriedade como substituta de
indicador de atendimento ao SUS.

O importador `app.tools.cnes_registry.import_cnes_csv` exige:
- `csv_path`: arquivo CSV extraído do recurso oficial.
- `database`: destino SQLite local.
- `columns`: mapeamento explícito dos onze campos lógicos para cabeçalhos reais;
  o campo `sus` deve comprovar vínculo ao SUS na fonte validada.
- `reference_date`: data de referência da exportação.
- `source_url`: URL HTTPS do recurso de onde os dados vieram.
- `delimiter`: separador do arquivo; padrão `;`.

Exemplo adaptável (substituir os nomes pelo cabeçalho real):

```python
from datetime import date
from pathlib import Path
from app.tools.cnes_registry import import_cnes_csv

columns = {
    "cnes": "CNES",
    "name": "NOME_FANTASIA",
    "type_code": "TP_UNID",
    "street": "LOGRADOURO",
    "number": "NUMERO",
    "district": "BAIRRO",
    "city": "MUNICIPIO",
    "state": "UF",
    "latitude": "LATITUDE",
    "longitude": "LONGITUDE",
    "sus": "ATENDE_SUS",
}
count = import_cnes_csv(
    Path("cnes.csv"),
    Path("data/cnes.sqlite"),
    columns,
    date(2026, 10, 7),
    "https://URL-REAL-DO-RECURSO.csv",
)
print(count)
```

**Os cabeçalhos no exemplo são ilustrativos**, não garantidos para o recurso oficial.

Códigos tratados: 01/02 → atenção básica; 73 → pronto atendimento;
05/07/20/21 → grupo hospital/pronto-socorro. Código 73 não comprova
que a unidade é UPA 24h; código 05 ou 07 não comprova pronto-socorro aberto.
O provedor local **não garante disponibilidade, horário, vagas ou atendimento
emergencial**. A importação rejeita linhas sem vínculo SUS afirmativo,
código CNES válido, nome, endereço e coordenadas no Brasil.

A criação é atômica e exclusiva: um banco existente não é substituído. Para
atualizar, use um novo nome de arquivo e altere a configuração após conferir
contagens e integridade.
O provedor usa a base local apenas quando configurada explicitamente com
`HEALTHFLOW_CNES_DATABASE=data/cnes.sqlite`. Caso contrário, a aplicação
continua usando o serviço OpenStreetMap.

**Importante:** cadastro CNES não prova disponibilidade operacional.
Não usar dados cadastrados como garantia de atendimento imediato.

## Uso pelo terminal

Crie um arquivo `cnes-columns.json` com os onze pares campo lógico/cabeçalho
da **exportação que você conferiu** (veja o dicionário no exemplo anterior).
Após extrair o CSV oficial e verificar o separador:

```bash
uv run python -m app.tools.import_cnes \
  --csv /caminho/para/cnes.csv \
  --mapping cnes-columns.json \
  --reference-date 2026-10-07 \
  --source-url https://URL-OFICIAL-DO-RECURSO-BAIXADO \
  --database data/cnes.sqlite
```

Os valores de data e URL acima são **exemplos a substituir pelo recurso real**.
Para o ZIP conferido em 07/10/2026, use:

```bash
curl -fL -o data/raw/cnes_estabelecimentos_csv.zip \
  https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/CNES/cnes_estabelecimentos_csv.zip
uv run python -m app.tools.import_cnes \
  --csv data/raw/cnes_estabelecimentos_csv.zip --official-establishments \
  --reference-date 2026-10-07 \
  --source-url https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/CNES/cnes_estabelecimentos_csv.zip \
  --database data/processed/cnes-2026-10-07-active.sqlite
```

Defina `HEALTHFLOW_CNES_DATABASE=data/processed/cnes-2026-10-07-active.sqlite` para
usar o retrato local na busca de proximidade. O campo
`CO_AMBULATORIAL_SUS=SIM` comprova apenas vínculo ambulatorial ao SUS;
estabelecimentos sem esse indicador ficam fora do retrato. `TP_UNIDADE=73`
identifica pronto atendimento, mas não comprova certificação UPA 24h.
`TP_UNIDADE=5/7/20/21` identifica hospitais e categorias afins, mas não
comprova pronto-socorro em operação. **Não use esses resultados como confirmação
de atendimento emergencial ou de vagas.**
No `.env` do backend, após importação bem-sucedida:

```env
HEALTHFLOW_CNES_DATABASE=data/cnes.sqlite
```

Reinicie o backend. Quando a configuração não é fornecida, a fonte continua
sendo o OSM. Se for fornecida mas o banco estiver ausente/corrompido,
a busca falha explicitamente em vez de inventar uma unidade.

**Revisão de segurança:** o importador mapeia 73 para a categoria técnica
`UPA` e 05/07/20/21 para `EMERGENCY_ROOM` porque essa é a enumeração
atual do protótipo. Os códigos oficiais 73 e 05/07 **não comprovam**
funcionamento de UPA 24h nem atendimento emergencial no hospital.
Essas categorias precisam de refinamento antes de indicar serviços a pacientes.
