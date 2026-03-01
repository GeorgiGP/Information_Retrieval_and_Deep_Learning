# Word2Vec Skip-gram с Negative Sampling и Квадратична Форма

## Имплементация на Модифициран Word2Vec Модел за Български Език

**Автор:** Георги Х. Лазов
**Факултетен номер:** 0MI0600299
**Курс:** Търсене и извличане на информация. Приложение на дълбоко машинно обучение
**Преподавател:** проф. Стоян Михов
**Семестър:** Зимен 2025/2026

---

## Резюме (Abstract)

Настоящата работа представя имплементация на модифициран Word2Vec Skip-gram модел с negative sampling за обучение на словни влагания (word embeddings) върху корпус от български публицистични текстове. Ключовата модификация спрямо стандартния Word2Vec е въвеждането на матрица W като параметър на квадратична форма в модела. Имплементирани са функции за изчисляване на загуба и градиенти както за единични наблюдения, така и за партидна обработка с използване на тензорни операции. Резултатите демонстрират успешно групиране на семантично свързани думи в двумерното пространство.

**Ключови думи:** Word2Vec, Skip-gram, Negative Sampling, Word Embeddings, Natural Language Processing, Bulgarian Language

---

## 1. Въведение

### 1.1 Мотивация

Словните влагания (word embeddings) са фундаментален компонент в съвременната обработка на естествен език. Те представляват думите като плътни вектори в непрекъснато пространство, като семантично подобни думи се разполагат близо една до друга.

### 1.2 Цели на проекта

1. Имплементиране на функции за семплиране с негативни примери
2. Изчисляване на функция на загуба и градиенти за модифициран Skip-gram модел
3. Оптимизирано партидно изчисление на градиенти с тензорни операции
4. Имплементация на стохастично спускане по градиента (SGD)
5. Обучение на модела върху български текстов корпус

### 1.3 Използван корпус

Моделът е обучен върху **Корпус от публицистични текстове за Югоизточна Европа**, предоставен от Института за български език към БАН:
- **Брой документи:** 35,337
- **Размер:** 7.9 MB
- **Език:** Български
- **Източник:** http://dcl.bas.bg/BulNC-registration/

---

## 2. Теоретична Основа

### 2.1 Word2Vec Skip-gram Модел

Word2Vec Skip-gram моделът цели да предвиди контекстните думи по дадена целева дума. За всяка двойка (целева дума w, контекстна дума c) моделът учи две матрици на влагания:
- **U** — матрица на влаганията за целеви думи (размерност V × M)
- **V** — матрица на влаганията за контекстни думи (размерност V × M)

където V е размерът на речника, а M е размерността на влаганията.

### 2.2 Модификация с Квадратична Форма

В настоящата имплементация е въведена допълнителна матрица **W** (размерност M × M) като параметър на квадратична форма:

$$t = v_c \cdot (W \cdot u_w) - q$$

където:
- $u_w$ е влагането на целевата дума
- $v_c$ е влагането на контекстната дума
- $W$ е матрицата на квадратичната форма
- $q$ е логаритъм от вероятността за negative sampling

### 2.3 Negative Sampling

Negative sampling е техника за апроксимиране на softmax функцията, която значително ускорява обучението. Вместо да се изчислява вероятност върху целия речник, се избират n негативни примери с вероятност:

$$P(w) \propto f(w)^{0.75}$$

където $f(w)$ е честотата на думата в корпуса.

### 2.4 Функция на Загуба

Функцията на загуба за едно наблюдение е бинарна cross-entropy:

$$J = -\sum_{i} \left[ \delta_i \log(\sigma(t_i)) + (1-\delta_i) \log(1-\sigma(t_i)) \right]$$

където:
- $\sigma(x) = \frac{1}{1+e^{-x}}$ е сигмоидна функция
- $\delta_i = 1$ за положителния пример (i=0), $\delta_i = 0$ за негативните

---

## 3. Имплементация

### 3.1 Функции за Семплиране (`sampling.py`)

#### 3.1.1 createSamplingSequence

Създава последователност за семплиране, в която всеки индекс на дума се среща пропорционално на $f(w)^{0.75}$:

```python
def createSamplingSequence(freqs):
    seq = []
    for i, freq in enumerate(freqs):
        count = round(freq ** 0.75)
        seq.extend([i] * count)
    return seq
```

**Сложност:** O(V × средна честота^0.75)

#### 3.1.2 noiseDistribution

Изчислява логаритмите от вероятностите за negative sampling:

```python
def noiseDistribution(freqs, negativesCount):
    freqs_adjusted = np.round(np.array(freqs) ** 0.75)
    probs = freqs_adjusted / np.sum(freqs_adjusted)
    q_noise = np.log(probs * negativesCount)
    return q_noise
```

### 3.2 Изчисляване на Градиенти (`grads.py`)

#### 3.2.1 lossAndGradient (единично наблюдение)

Изчислява загубата и градиентите за едно наблюдение:

```python
def lossAndGradient(u_w, Vt, W, q):
    t = Vt @ (W @ u_w) - q
    sigma_t = sigmoid(t)

    delta_c = np.zeros_like(q)
    delta_c[0] = 1.0

    J = -np.sum(delta_c * np.log(sigma_t) +
                (1 - delta_c) * np.log(1 - sigma_t))

    diff = sigma_t - delta_c

    du_w = W.T @ (Vt.T @ diff)
    dVt = diff[:, np.newaxis] @ (u_w[np.newaxis, :] @ W.T)
    dW = (Vt.T @ diff[:, np.newaxis]) @ u_w[np.newaxis, :]

    return J, du_w, dVt, dW
```

**Градиенти:**
- $\frac{\partial J}{\partial u_w} = W^T V^T (\sigma(t) - \delta)$
- $\frac{\partial J}{\partial V} = (\sigma(t) - \delta) \cdot (u_w W^T)$
- $\frac{\partial J}{\partial W} = V^T (\sigma(t) - \delta) \cdot u_w^T$

#### 3.2.2 lossAndGradientBatched (партидна обработка)

Оптимизирана версия с тензорни операции за обработка на S наблюдения наведнъж:

```python
def lossAndGradientBatched(u_w, Vt, W, q):
    S, M = u_w.shape

    W_u = u_w @ W.T
    t = np.einsum('snm,sm->sn', Vt, W_u) - q

    sigma_t = sigmoid(t)

    delta_c = np.zeros_like(q)
    delta_c[:, 0] = 1.0

    allJ = -np.sum(delta_c * np.log(sigma_t) +
                   (1 - delta_c) * np.log(1 - sigma_t))
    J = np.sum(allJ) / S

    diff_S = (sigma_t - delta_c) / S

    du_w = np.einsum('snm,sn->sm', Vt, diff_S) @ W
    dVt = np.einsum('sn,sm->snm', diff_S, u_w @ W.T)
    dW = np.einsum('snm,sn,sk->mk', Vt, diff_S, u_w)

    return J, du_w, dVt, dW
```

**Ключови оптимизации:**
- Използване на `np.einsum` за ефективни тензорни контракции
- Избягване на експлицитни цикли
- Векторизирани операции за цялата партида

### 3.3 Стохастично Спускане по Градиента (`w2v_sgd.py`)

```python
def stochasticGradientDescend(data, U0, V0, W0, contextFunction,
                               lossAndGradientFunction, q_noise,
                               batchSize=1000, epochs=1, alpha=1.):
    U, V, W = U0, V0, W0
    idx = np.arange(len(data))

    for epoch in range(epochs):
        np.random.shuffle(idx)
        for b in range(0, len(idx), batchSize):
            # Подготовка на партидата
            batchData = [(w, contextFunction(c))
                         for w, c in data[idx[b:b+batchSize]]]

            # Изчисляване на градиенти
            J, du_w, dVt, dW = lossAndGradientFunction(u_w, Vt, W, q)

            # Актуализация на параметрите
            W -= alpha * dW
            for k, (w, context) in enumerate(batchData):
                U[w] -= alpha * du_w[k]
                V[context] -= alpha * dVt[k]

    return U, V, W
```

---

## 4. Параметри на Модела

| Параметър | Стойност | Описание |
|-----------|----------|----------|
| embDim | 50 | Размерност на влаганията |
| windowSize | 3 | Размер на контекстния прозорец |
| negativesCount | 5 | Брой негативни примери |
| batchSize | 1000 | Размер на партидата |
| epochs | 1 | Брой епохи |
| alpha | 1.0 | Скорост на обучение |
| vocabularySize | 20,000 | Размер на речника |

---

## 5. Резултати

### 5.1 Визуализация на Влаганията

След обучението, влаганията са редуцирани до 2D чрез SVD и нормализирани. На фигурата по-долу е показано разполагането на избрани думи:

![Word Embeddings Visualization](a2/embeddings.png)

### 5.2 Семантично Групиране

Визуализацията демонстрира успешно семантично групиране:

**Група 1: Времеви понятия**
- "януари", "октомври" — месеци, групирани в горната лява част

**Група 2: Икономически термини**
- "пазар", "стоки", "бизнес", "фирма", "бюджет" — икономически понятия, групирани в долната дясна част

**Група 3: Енергийни ресурси**
- "петрол", "нефт" — синоними, разположени близо един до друг

### 5.3 Производителност

Партидната версия (`lossAndGradientBatched`) постига **над 2 пъти по-бързо изпълнение** в сравнение с кумулативната версия благодарение на:
- Векторизирани тензорни операции
- Ефективно използване на `numpy` и `einsum`
- Избягване на Python цикли

---

## 6. Тестове

Всички имплементирани функции са валидирани чрез предоставените тестове:

| Тест | Функция | Резултат |
|------|---------|----------|
| test 3 | createSamplingSequence | ✓ Преминат |
| test 3 | noiseDistribution | ✓ Преминат |
| test 4 | lossAndGradient | ✓ Преминат |
| test 5 | lossAndGradientBatched | ✓ Преминат |
| test 6 | stochasticGradientDescend | ✓ Преминат |

---

## 7. Изводи

### 7.1 Постигнати резултати

1. **Успешна имплементация** на модифициран Word2Vec Skip-gram модел с квадратична форма
2. **Ефективно партидно изчисление** на градиенти с тензорни операции
3. **Семантично смислени влагания** за български език
4. **Валидация** чрез всички предоставени тестове

### 7.2 Възможности за развитие

- Увеличаване на размерността на влаганията (100-300)
- Експерименти с различни стойности на negativesCount
- Прилагане на learning rate scheduling
- Оценка чрез word analogy задачи

---

## 8. Технически Детайли

### 8.1 Изисквания

```
Python >= 3.5
numpy
nltk
matplotlib
scikit-learn
```

### 8.2 Структура на Проекта

```
HW2/
├── README.md                 # Настоящият документ
├── Tasks_1_and_2.pdf        # Теоретични задачи
├── a2/
│   ├── grads.py             # Функции за загуба и градиенти
│   ├── sampling.py          # Функции за семплиране
│   ├── w2v_sgd.py           # SGD имплементация
│   ├── utils.py             # Помощни функции
│   ├── run.py               # Основен скрипт за обучение
│   ├── test.py              # Тестове
│   ├── w2v-U.npy            # Обучена U матрица
│   ├── w2v-V.npy            # Обучена V матрица
│   ├── w2v-W.npy            # Обучена W матрица
│   └── embeddings.png       # Визуализация
└── FN0MI0600299/            # Финална версия за предаване
```

### 8.3 Изпълнение

```bash
# Активиране на средата
conda activate tii

# Изпълнение с партидни градиенти (по подразбиране)
python run.py

# Изпълнение с кумулативни градиенти
python run.py cumulative

# Изпълнение на тестове
python test.py 3  # Тест за sampling функции
python test.py 4  # Тест за lossAndGradient
python test.py 5  # Тест за lossAndGradientBatched
python test.py 6  # Тест за SGD
```

---

## Референции

1. Mikolov, T., et al. (2013). *Efficient Estimation of Word Representations in Vector Space*. arXiv:1301.3781
2. Mikolov, T., et al. (2013). *Distributed Representations of Words and Phrases and their Compositionality*. NIPS 2013
3. Goldberg, Y., & Levy, O. (2014). *word2vec Explained: Deriving Mikolov et al.'s Negative-Sampling Word-Embedding Method*. arXiv:1402.3722
4. Корпус от публицистични текстове за Югоизточна Европа, Институт за български език, БАН

---

*Документът е създаден като част от Домашно задание 2 по курса "Търсене и извличане на информация. Приложение на дълбоко машинно обучение", ФМИ, СУ "Св. Климент Охридски", 2025/2026*
