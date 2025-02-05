# TP03

### Execution `laplace3d.cu`

On remarque une accelération bien plus importante lors de l'utilisation du GPU qui prend ~25ms, face au CPU qui prend ~4000ms. <br>

#### Kernel
Le kernel calcul l'approximation de l'équation pour la case de la grille u1 du thread. <br>
Sous les conditions : 
    - Le thread se trouve dans la grille
    - Le thread ne se trouve pas au bord de la grille

#### Main
On initialise la grille qu'on copie dans la mémoire `device`. Puis on lance les blocs de threads un nomreb `REPEAT` de fois. À chaque fois on swap la grille input et output afin de recalculer la grille à chaque fois.

