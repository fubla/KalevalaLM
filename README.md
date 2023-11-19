# KalevalaLM
Tiny decoder-only transformer model trained with the Finnish National Epic - Kalevala. Generates pseudo-Finnish poetry in the Kalevala style.

## Description
Comes with two pre-trained models: 

`kalevala-char.pt` which uses a character tokenizer 

Sample:
```
"Siellä nyt, hyvy lolonua,
niiskiset käken kiskupti;
muriksikkö maallenki
nuoren nousuttelevi,
pulki pään kätteheksi.
Siitä kähin keikkahun,
kutitihin kukkulut niemoien
oman kukkuleman mäelle,
riipukut voitalle,
kaikkimmenna kaotille;
vieri ei korvesten kohot,
miehen kolmettomaksi
yntymän välisnytellen,
kuulunna kulelonna!"
Sanoi emon sanoiksi:
"Vaka vanha Väinämöinen
utuikse tuon puolihiksi,
hope'iksi luotelluksi,
kauen kankka kaunis kaunihin,
tuperi tulennut tutumahan,
maaman maatoni sinulhut,
tuvanokonut kuolmisekkonna.
"Sulho, jäätä Jouka,
rautoi ongen tuulta,
hauset haukih juotua,
vennoset kaikken suusihin,
metosen jotehet levesi,
seka keski sinnalla kuhkahui..
Piitä poika pololnensa,
korotin kellartojansa.
Sanoi satki virkkoi noinen,
saatset naisen imestänsä
maatonut miekakkana,
panioturin palvikkana.
Ammu ukuttelevi,
kasteleiten, käärtelevi,
kun katseleinevi.
```


and 

`kalevala-p50k.pt` which uses the [OpenAI Tiktoken tokenizer](https://github.com/openai/tiktoken)

Sample:
```
Kun linnun, maariolan,
saa mukahan-alainen Jumalan,
verkaltoin nuoren neitoin:
puhme joukasia kaimohon,
havaturpoimi Tuonen mustahan,
lähängen saattomahan?"
Mit' on lieto Lemminkäinen.
Itse kuun lausui,
itse vanha rylyn noian:
"Saat saatanehe, soita,
liekunne surmaiset tieriset?"
Hiien häytänpä surolikomahan,
kuihä tieto Lemminkäinen
istarsan Antero pojan pajahan
juoa tappeloita;
konsa soitasi;
etkävältä kolmen on sopuansa,
äsäänti kanteloita;
isutu kuullista kaheksi,
siihen Pohjan kuolen.
Itse seppo Ilmarinen
nosta arvelee, tuon tuon merellot.
Kullervo, Kalervon poika,
vasistoi siuvoa Kalervon,
tuosta alemastahan:
"Ei toinen tehtohon poika,
lähtoa ukahan kultaturkkahan
päähän utuisen!
Untamo suku-ikuinen:
"Miks'Kysyvä minä
mutkan makohon jälksehen;
nauralle jään sisähän unehen,
vähäärän suuren laukin,
elinajan kannasjollehen,
siellen meren juottihe,
leukan kypärmehen,
leuo'ito nuoren juuren.
Itse lentälehen itkselle,
sattini itse tietäjän.
```

## Usage
### Generate
```
python model.py [model-file] [--tokenizer=char|tiktoken]
```

## Training
```
python train.py [training-file] [output-model-file] [--tokenizer=char|tiktoken] [--stop-early-steps]
```
