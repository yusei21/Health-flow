# Importação local CNES (experimental)

Fonte: [CNES, Dados Abertos do Ministério da Saúde](https://dadosabertos.saude.gov.br/dataset/cnes-cadastro-nacional-de-estabelecimentos-de-saude).

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
    "cnes": "CNES", "name": "NOME_FANTASIA", "type_code": "TP_UNID",
    "street": "LOGRADOURO", "number": "NUMERO", "district": "BAIRRO",
    "city": "MUNICIPIO", "state": "UF",
    "latitude": "LATITUDE", "longitude": "LONGITUDE", "sus": "ATENDE_SUS",
}
count = import_cnes_csv(
    Path("cnes.csv"), Path("data/cnes.sqlite"), columns,
    date(2026, 10, 7), "https://URL-REAL-DO-RECURSO.csv",
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

A atualização é atômica: falha de validação não substitui a base anterior.
O provedor usa a base local apenas quando configurada explicitamente com
`HEALTHFLOW_CNES_DATABASE=data/cnes.sqlite`. Caso contrário, a aplicação
continua usando o serviço OpenStreetMap.

**Importante:** cadastro CNES não prova disponibilidade operacional.
Não usar dados cadastrados como garantia de atendimento imediato.
