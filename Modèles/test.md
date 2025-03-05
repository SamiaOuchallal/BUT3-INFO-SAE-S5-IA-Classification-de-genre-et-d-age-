| Critère                | Adam                          | RMSprop                      | SGD                          |
|------------------------|-------------------------------|------------------------------|------------------------------|
| **Taux d'apprentissage** | Adaptatif                     | Adaptatif                    | Fixe (nécessite réglage)     |
| **Convergence**         | Rapide                        | Modérée                      | Lente (sans momentum)        |
| **Mémoire utilisée**    | Élevée (stocke 2 moments)     | Modérée (stocke 1 moment)    | Faible                       |
| **Robustesse**          | Très robuste                  | Robuste                      | Sensible au réglage          |
| **Utilisation typique** | Problèmes complexes           | Problèmes intermédiaires     | Problèmes simples ou contrôle fin |
