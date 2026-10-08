# Protocolo de avaliação experimental do Health-flow

O benchmark precisa responder a duas perguntas separadas: como cada classificador
se comporta com os dados disponíveis e se o Harness respeita suas regras durante
o encaminhamento. Os JSONs de treino respondem apenas à primeira pergunta.

## Conjuntos de dados e estimando

O experimento `synthetic_baseline` mede quanto os modelos reproduzem regras
do gerador sintético. Não mede segurança ou eficácia em pacientes.

O `structured_mimic_baseline` usa sinais vitais da triagem do pronto atendimento
dos Estados Unidos. Os níveis ESI são mapeados experimentalmente para três rótulos.
Esse mapeamento não equivale ao protocolo de encaminhamento do SUS. A API atual
não recebe esses sinais vitais, portanto o modelo MIMIC não pode substituí-la
diretamente. Um treino e teste no próprio MIMIC avalia generalização dentro da
amostra; não é um teste entre domínios.

Para cada execução, registre versão da fonte, regras de exclusão, contagem de linhas
e pacientes, mapeamento de rótulos, proporção de faltantes, versão do código e do
scikit-learn. Não publique dados clínicos identificáveis.

## Partições e seleção

O projeto separa aproximadamente 20% para teste. Quando há identificador de
paciente, pacientes de treino e teste não se sobrepõem. Os candidatos são
comparados usando validação cruzada somente no treino. O critério atual prioriza
recall da classe `EMERGENCY` arredondado a duas casas e usa F1 macro como desempate.
O teste reservado serve para relatar desempenho do modelo já selecionado.

A partir desta revisão, o treinamento externo **recusa** um experimento de três
classes se alguma classe tiver menos de dois exemplos no teste ou em qualquer
partição de validação da CV. Dois exemplos ainda são insuficientes para uma
inferência forte; esse é um mínimo de integridade, não uma meta amostral.
Se a divisão por paciente deixar uma classe ausente, aumente o dataset ou
redefina previamente o desenho. Não replique registros, invente rótulos nem
altere o random seed repetidamente até obter uma divisão favorável.

## Auditoria dos registros

Para a execução da amostra MIMIC Demo já registrada:

```bash
make benchmark-audit AUDIT_RUN=781fd757
```

Para a execução sintética de referência:

```bash
make benchmark-audit AUDIT_RUN=eb0307e7
```

O arquivo `benchmarks/results/audit_<run_id>.json` apresenta os quatro modelos,
identifica a seleção feita por CV, confere somas e suportes da matriz de confusão,
aponta classes ausentes, quantifica emergências subestimadas e calcula o intervalo
de Wilson de 95% para o recall de emergência. O intervalo considera registros
como independentes: **não** é ajustado para agrupamento por paciente e deve ser
tratado como descrição exploratória em bases com visitas repetidas.

A amostra MIMIC-IV-ED Demo 2.2 tinha 207 exemplos após limpeza, dos quais apenas
dois pertenciam à classe `PRIMARY_CARE`. O teste reservado não continha essa
classe. Os resultados históricos permanecem preservados; uma nova execução com o
protocolo mais rigoroso recusará esse experimento, em vez de publicar F1 macro
como se as três classes tivessem sido avaliadas.

## Comparações e limitações

Relate matriz de confusão, suporte por classe, F1 macro, recall de emergência,
under-triage e critical under-triage. Compare latência apenas em condições
documentadas de hardware, execução e aquecimento. Não interprete diferenças
entre sintético e MIMIC como uma medição isolada de domain shift: mudam as
features, as populações e a definição dos rótulos.

Para estudos confirmatórios, é necessário um conjunto com cobertura suficiente,
preferencialmente uma coorte externa, intervalos ajustados ao agrupamento por
paciente, comparação pareada dos modelos na mesma amostra e repetição com
seeds previamente fixadas. Calibração, análise por subgrupos quando permitido,
ablação do contexto do paciente e avaliação de cobertura geográfica também
estão pendentes.

## Avaliação do Harness: trabalho separado

O classificador prediz uma classe; o Harness combina extração, contexto,
regras de segurança e disponibilidade de unidades. Avaliar o modelo sozinho
não determina o erro de encaminhamento final. Um benchmark de ponta a ponta
precisa de relatos com rótulos independentes, fontes autorizadas e resultados
referenciais compatíveis com o problema brasileiro. Deve registrar ainda:
falhas de LLM e ferramenta, fallback, quantidade de chamadas, latência p50/p95,
respeito às invariantes e alterações no resultado provocadas pelo contexto.
Os testes automatizados da política verificam comportamento de software,
não são substitutos de validação clínica.

O Jev não participa da execução principal nem deste protocolo.
