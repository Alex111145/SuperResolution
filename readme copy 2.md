# 1. Pulisci installazioni precedenti
rm -rf astap_local temp_db astap_old.deb d50_star_database.deb

# 2. Scarica ASTAP versione 2022 (Compatibile con il tuo server)
wget -O astap_old.deb "https://web.archive.org/web/20221028194951if_/https://www.hnsky.org/astap_amd64.deb"

# 3. Scarica il Database delle stelle (D50)
wget -O d50_star_database.deb "https://sourceforge.net/projects/astap-program/files/star_databases/d50_star_database.deb/download"


# 1. Estrai il programma ASTAP
dpkg -x astap_old.deb astap_local

# 2. Dai i permessi di esecuzione al file del programma
chmod +x astap_local/opt/astap/astap

# 3. Estrai il Database in una cartella temporanea
dpkg -x d50_star_database.deb temp_db

# 4. Sposta il Database accanto al programma
# (Sposta tutto il contenuto da temp_db/opt/astap dentro la tua cartella locale)
mv temp_db/opt/astap/* astap_local/opt/astap/

# 5. Rimuovi la cartella temporanea
rm -rf temp_db

./astap_local/opt/astap/astap -f -v