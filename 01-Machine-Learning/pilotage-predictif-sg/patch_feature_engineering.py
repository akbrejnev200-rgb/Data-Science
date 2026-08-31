"""
Ce script modifie feature_engineering.py pour sauter l'ecriture de features_detail dans Neon.
Place ce fichier dans C:\\Users\\akbre\\Downloads\\python\\mon_projet et lance-le.
"""
import os

# Chemin vers le fichier a modifier
chemin = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "notebooks_avec_ecriture_dans_database",
    "feature_engineering.py"
)

# Lire le fichier
with open(chemin, "r", encoding="utf-8") as f:
    contenu = f.read()

# Chercher la ligne qui ecrit features_detail
ancienne = 'df.to_sql("features_detail", engine, if_exists="replace", index=False, chunksize=5000)'
nouvelle = '# SKIP: features_detail trop volumineux pour Neon free plan (512 MB)\n    # df.to_sql("features_detail", engine, if_exists="replace", index=False, chunksize=5000)\n    print("  features_detail : IGNORE (trop volumineux pour Neon free plan)")'

if ancienne in contenu:
    contenu = contenu.replace(ancienne, nouvelle)
    with open(chemin, "w", encoding="utf-8") as f:
        f.write(contenu)
    print("feature_engineering.py modifie avec succes.")
    print("La table features_detail ne sera plus ecrite dans Neon.")
else:
    # Essayer avec des guillemets simples
    ancienne2 = "df.to_sql('features_detail', engine, if_exists='replace', index=False, chunksize=5000)"
    if ancienne2 in contenu:
        nouvelle2 = "# SKIP: features_detail trop volumineux pour Neon free plan (512 MB)\n    # df.to_sql('features_detail', engine, if_exists='replace', index=False, chunksize=5000)\n    print('  features_detail : IGNORE (trop volumineux pour Neon free plan)')"
        contenu = contenu.replace(ancienne2, nouvelle2)
        with open(chemin, "w", encoding="utf-8") as f:
            f.write(contenu)
        print("feature_engineering.py modifie avec succes.")
    else:
        print("ATTENTION : ligne non trouvee automatiquement.")
        print("Ouvre manuellement le fichier feature_engineering.py")
        print("et commente la ligne 301 qui contient : df.to_sql('features_detail'...)")
