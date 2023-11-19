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
"Ota korjama mansikota pitu,
kai ilman nuoren vännitäjät,
kyestä lienotiset!
Tee täällä tippoa,
attehet näähän saata,
ei kultainen kainen
ennen käättyä äänen,
rakehille turme'ille lepehiksi,
hukkujoka lajuksi,
oluoret nuori näkemiä,
hämän kaunihistä,
ajoi suun eläiltä!
"Hyvät sinä parempia,
nälkä minua venehen,
taivustan solmman,
josin rauan minun alla näkevi,
laske pahoin pääiveä."
Louhi, tuota tunsi, tuo sanoiksi virkki:
"Mehilmäinen, itse, pitkin ja!
Tore siellut laulu,
Lemmon sinulta pääni
pienet suotta juotamatta,
otipuvon juoteli,
kauleppyräpasehen sysämpehät,
pää oli palkojasta parekkelästä,
päällä päiväsi polven,
jäisen joukkuhunun juoksi.
"Mene en huasi sukkulehen,
su laulle jaloahensa
Hiien rannan rupele,
yhen pituo'ille,
puisivet juoa riehet,
Ain miehet mointa juomaltani,
oikki uut terverehet,
vasellos vallan."
Louhi, akka eikki lapsi,
hursipi tuotakoiksi,
vammi hauolle helissansa.
Kaaloin, niin kertoelinki
alla pahoilla mielin,
perin menevyränsä.
Sanan virkkoi, noin nimesi:
"Lauloi minulle kuolle,
niin parempi minunen etehen,
suovan ei sukkua,
kaikki tyttöä vaimo?"
Laski lieto Lemminkäinen:
"Jos ei se sauoma,
souta luotiminen pursi,
niillä toista venehen,
impyhän ulvahan."
Toivi siivatki sanoa;
sanan virkkoi, noin nimesi:
"Akka ume, et lienet laulamme,
parempi laulan lepimenki
paremmin poisijalta sisujemme."
Vaka vanha Väinämöinen
sanan virkkoi, noin nimesi:
"Voi minosta toisi, tuleltani
tietämä päivällä,
kulkevi jalkojanlainen!
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
