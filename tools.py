
# coding: utf-8

# In[23]:


import re 
import glob
import json

def ecrire(donnees, chemin):
  w = open(chemin, "w")
  w.write(donnees)
  w.close()

def lire_json(path):
  f = open(path)
  dic = json.load(f)
  f.close()
  return dic

def get_all_files(dic):
  liste_all_files = []
  for dataset, classes in dic.items():
    for cl, liste_files in classes.items():
      liste_all_files+=liste_files
  return liste_all_files

def sauvegarder(donnees, chemin):
  donnees_json=json.dumps(donnees, indent=2)
  ecrire(donnees_json, chemin)

def nettoyer(chaine):
  chaine = re.sub(";|'|:|\.|\n|\r|\(|\)|\"|\?|\!", " ", chaine)
  return chaine.lower()

def stats(chaine):
  dic = {}
  for car in chaine:
    dic.setdefault(car, 0)
    dic[car]+=1
  return dic

def get_stats_longueur(liste):
  longueurs = []
  for item in liste:
    longueurs.append(len(item))
  return {"L_moyenne_mots":get_moyenne(longueurs, 2), "L_min_mots": min(longueurs), "L_max_mots":max(longueurs)}

def get_moyenne(liste, arrondi=5):
  total = 0
  for chiffre in liste:
    total+=chiffre
  return round(float(total)/len(liste), arrondi)

def get_effectif_adverbes(mots):
  liste_adverbes = lire_json("liste_adverbes_en.json")
  dic_adverbes = {x:0 for x in liste_adverbes}
  dic_adverbes["total_adverbes"] = 0
  for m in mots:
    if m in dic_adverbes:
      dic_adverbes[m]+=1
      dic_adverbes["total_adverbes"] +=1
  return dic_adverbes

def get_stats_caracteres(chaine_nettoyee):
  dic_stats_caracteres = stats(chaine_nettoyee)
  stats_triee_caracteres = trier_dic(dic_stats_caracteres)
  return stats_triee_caracteres

def get_stats_mots(liste_mots):
  dic_stats_mots=stats(liste_mots)
  stats_triee_mots = trier_dic(dic_stats_mots)
  return stats_triee_mots

def trier_dic(dic):
  l = [ [y,x]  for x,y in dic.items()]
  l_triee = sorted(l, reverse=True)
  return [[x,y] for y,x in l_triee]

def get_types_phrases(chaine):
  nb_affirm = len(re.findall("\. ", chaine))
  nb_interro = len(re.findall("\? ", chaine))
  nb_excla = len(re.findall("\! ", chaine))
  return {"nb_affirm":nb_affirm, "nb_interro":nb_interro, "nb_excla":nb_excla}


def combien_de(fichier):
    happy=0
    sad=0
    neutre=0
    dicto_de_words=lire_json("liste_emotions_en.json")
    words = [w.lower() for w in fichier]
    for x in words:
        if x in dicto_de_words["happy and positive"]:
            happy+=1
        if x in dicto_de_words["sad and negative"]:
            sad+=1
        if x in dicto_de_words["neutral and object"]:
            neutre+=1
    percentage_happy = round(happy/len(words),5)
    percentage_sad = round(sad/len(words),5)
    percentage_neutre = round(neutre/len(words),5)
    return{"nb_happy":happy,"nb_sad":sad,"nb_neutre":neutre,"pourcentageHappy":percentage_happy,"pourcentageSad":percentage_sad,"pourcentageNeutre":percentage_neutre}

