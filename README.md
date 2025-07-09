# AI Challenge: CV2JD Smart Match
Qëllimi i kësaj sfide është të zhvilloni një prototip të një sistemi të bazuar në AI që analizon CV-të dhe i përputh ato me përshkrimet e punës (JDs) në bazë të përmbajtjes, kontekstit dhe përputhjes së aftësive.

## Të dhënat që do përdoren:
### Dataseti i CV-ve (2400+ CV):

 - ID – ID e dokumentit fizik të CV-së
 - Resume_str – Teksti i përpunuar nga formati PDF
 - Resume_html – Përmbajtja në HTML e shkëputur nga PDF
 - Category – Kategoria e punës për të cilën ka aplikuar kandidati (p.sh., Teknologji Informacioni, Shëndetësi, Bankë, Ndërtim, etj.)
 
 ### Dataseti i Përshkrimeve të Punës (850+ JD):

 - company_name – Emri i kompanisë që ka postuar JD-në
 - job_description – Përmbajtja e përshkrimit të punës
 - position_title – Titulli i pozicionit të shpallur
 - description_length – Gjatësia në karaktere e përshkrimit
 - model_response – Përmbledhje nga model AI në format JSON

## Struktura e Sfidës
Gjithsejt janë 3 detyra me vështirësi të ndryshme:

 - Eksplorimi dhe përpunimi i të dhënave (EDA)
 - Modelimi i një përputhje mes CV/JD
 - Krijimi i një modeli për vlerësimin dhe rankimin e talenteve

Komplet sfida është planifikuar të marrë deri 10 orë dhe pritet të përfundohet në një periudhë prej 2 javësh (pas shpalljes).

## Detyra 1: Eksplorimi dhe Përpunimi i të Dhënave (EDA)
__Niveli:__ Fillestar
__Objektiva:__

 - Analizoni dhe eksploroni të dhënat nga CV-të dhe JD-të.
 - Kryeni pastrimin dhe përgatitjen bazë të tekstit.
 - Nxirrni karakteristika bazë nga përmbajtja tekstuale.

### Çfarë pritet të dorëzoni:

 - Analizë statistikore e kategorive të CV-ve dhe përshkrimeve të punës.
 - Pastrim i tekstit (heqja e shenjave të pikësimit, stopwords, karaktere jo-ASCII).
 - Tokenizim, lematizim.
 - Vizualizime të fjalëve më të shpeshta për çdo kategori pune.

### Teknologjitë e rekomanduara:
Python/Jupyter Notebooks (Pandas, NLTK/SpaCy), Matplotlib/Seaborn/Plotly

## Detyra 2: Model për Përputhje CV ⇄ JD
__Niveli:__ Mesatar
__Objektiva:__

 - Përdorni teknikë të ngjashmërisë (TF-IDF, Sentence Embeddings) për të krahasuar përmbajtjen.
 - Ndërtoni një sistem që përputh çdo CV me JD-të më të përshtatshme.

### Çfarë pritet të dorëzoni:

 - Implementim i TF-IDF ose embeddings për të konvertuar tekstin.
 - Llogaritje e ngjashmërisë (p.sh., cosine similarity).
 - Kthim i Top 5 JD-ve më të përshtatshme për çdo CV ose për një CV të dhënë.
 - Vizualizim i rezultateve të ngjashmërisë (matricë ose heatmap).
 - Analizë krahasuese mbi cilësinë e përputhjes.

### Teknologjitë e rekomanduara:
Scikit-learn, sentence-transformers, NumPy, Matplotlib

## Detyra 3: Model për Vlerësimin dhe Rankimin e Talenteve
__Niveli:__ Avancuar
__Objektiva:__

 - Zhvilloni një model që vlerëson përputhjen e një CV-je me një JD të caktuar në bazë të kërkesave të nxjerra nga model_response.

### Çfarë pritet të dorëzoni:

 - Nxjerrje e fushave si “Required Skills”, “Educational Requirements”, etj., nga JD JSON.
 - Ndërtim i karakteristikave për krahasim (strukturor dhe tekstual).
 - Zhvillim i një modeli për vlerësim (p.sh., klasifikues ose sistem rankimi).
 - Output: një pikë përputhjeje 0–100 për secilën CV ndaj një JD.
 - Vlerësim i performancës me metrika si Precision@K, nDCG ose ROC-AUC.

### Teknologjitë e rekomanduara:
Hugging Face Transformers, Scikit-learn, XGBoost, NLP embeddings, LLM-ve

# Kriteret e Vlerësimit

 - Prezentimi i të Dhënave -20%
 - Cilësia e Kodit - 10%
 - Performanca e Modelit - 45%
 - Krijimtaria/Inovacioni - 15%
 - Prezantimi i Punës - 10%

## Këshilla

 - Përdorni një qasje eksperimentale me prova dhe gabime.
 - Pjesa e avancuar është fleksibile – mund të përdorni modele klasifikimi, rankimi ose rregulla të personalizuara.
 - Nxitni kreativitetin dhe analizën përtej zgjidhjeve standarde.