#import "@preview/polylux:0.4.0": *
#import "@preview/diagraph:0.3.6": *

#set page(paper: "presentation-16-9")
#set text(font: "Liberation Sans", size: 20pt)

#let rustconf-theme(body) = {
  set page(
    fill: rgb("#1a1a1a"),
    margin: (x: 2em, y: 2em),
    footer: align(right)[
      #text(size: 14pt, fill: rgb("#808080"))[
        #context counter(page).display("1 / 1", both: true)
      ]
    ],
  )
  set text(fill: rgb("#f0f0f0"))

  show heading.where(level: 1): it => {
    set text(size: 32pt, weight: "bold", fill: rgb("#ff6b35"))
    it
  }

  show heading.where(level: 2): it => {
    set text(size: 24pt, weight: "bold", fill: rgb("#f7931e"))
    it
  }

  show raw.where(block: true): it => {
    set text(size: 14pt, font: "Fira Code")
    block(
      fill: rgb("#2d2d2d"),
      inset: 1em,
      radius: 0.5em,
      width: 100%,
      it
    )
  }

  show raw.where(block: false): it => {
    box(
      fill: rgb("#2d2d2d"),
      inset: (x: 0.3em, y: 0.1em),
      radius: 0.2em,
      it
    )
  }

  body
}

#show: rustconf-theme

// Title Slide
#slide[
  #align(left + horizon)[
    #text(size: 72pt, weight: "bold", fill: rgb("#ff6b35"))[
      Morok
    ]

    #v(1em)

    #text(size: 22pt, fill: rgb("#d0d0d0"))[
      Минималистичный DL фреймворк на Rust
    ]

    #v(3em)

    #text(size: 20pt, fill: rgb("#c0c0c0"))[
      RustCon 2025
    ]

    #v(0.5em)

    #text(size: 18pt, fill: rgb("#a0a0a0"))[
      Пацакула Никита
    ]
  ]
]

#slide[
  = Обо мне

  #set text(size: 22pt)

  #v(1em)

  #list(
    [Пишу на Rust последние 6 лет],
    [Работал над сложными проектами
      #list(
        [Отечественная СУБД],
        [Web-scale поисковик],
        [Web-scale ML-пайплайны]
      )
    ],
    [Последние полтора года работаю CTO в Cognito]
  )
]

#slide[
  = Cognito

  #set text(size: 18pt)

  #one-by-one[
    - Разработка аналитических систем
  ][
    - Сбор открытой информации: \~1400 PPS, картинки, видео, аудио
  ][
    - ML пайплайн
      - Анализ текста: PoS, NER, Sentence segmentation, Topic segmentation
      - Анализ аудио: транскрибация, идентификация говорящих, Intent analysis
      - Поиск видео: поиск похожих, поиск частей по видео, поиск видео по части
  ][
    - Поиск по тексту
      - Семантический: ищем близкое по смыслу
      - По теме: ищем сообщения по той же теме, что исходное
      - Полнотекстовый
  ][
    - Rust инфраструктура
      - 1.2 миллиона строк кода на Rust
      - Не на Rust: полнотекстовый поиск, DL фреймворки
  ]
]

#slide[
  = DL фреймворк
  #enum(
    [Работают с многомерными матрицами (Тензорами)],
    [Позволяет выполнять на ней ограниченный, но достаточный набор операций],
    [Использует аппаратное ускорение (GPU, TPU) для ускорения вычислений],
    [Предоставляет частоиспользуемые функции, которые нужны для DL]
  )
]

// Problems Slide
#slide[
  = Проблемы крупных ML фреймворков

  #v(1em)

  #set text(size: 24pt)

  #enum(
    numbering: "1.",
    [Потребность работать в зоопарке аппаратного ускорения],
    [Сложно вносить изменения и профилировать],
    [Сложная и дорогая DevOps/MLOps инфраструктура],
    [Небольшая площать интеропа],
    [Невозможность использовать большую часть экосистемы]
  )
]

// Frameworks Comparison - Overview
#slide[
  = ML фреймворки: история

  #v(2em)

  #align(center)[
    #image("frameworks_histogram.svg", width: 95%)
  ]
]

// Framework Popularity Trends
#slide[
  = ML фреймворки: популярность

  // #v(1em)

  #align(center)[
    #image("frameworks_trends.png", width: 95%)
  ]
]

// PyTorch Architecture
#slide[
  = Архитектура PyTorch

  #v(1em)

  #grid(
    columns: (1fr, 1fr, 1fr, 1fr, 1fr),
    gutter: 0.8em,
    align: center,
    [
      *Python API*\
      Tensor operations\
      Autograd\
      #sym.arrow.b\
      ATen operators
    ],
    [
      *Dispatcher*\
      Dynamic dispatch\
      Device selection\
      #sym.arrow.b\
      Backend kernels
    ],
    [
      *Autograd Engine*\
      Build computation\
      graph\
      Backward pass\
      #sym.arrow.b\
      Gradients
    ],
    [
      *ATen*\
      Operator library\
      CPU/CUDA kernels\
      #sym.arrow.b\
      Optimized code
    ],
    [
      *Backends*\
      cuDNN/cuBLAS\
      MKL/oneDNN\
      #sym.arrow.b\
      Hardware execution
    ]
  )
]

#slide[
  = PyTorch: Что не так

  #set text(size: 20pt)

  #grid(
    columns: (1fr, 1fr),
    gutter: 2em,
    one-by-one[
      == Аппаратная поддержка

      - Быстро работает только Nvidia
      - Хочешь добавить свой ускоритель -- реализуй 250 ядер и 50 операторов
      - В upstream всё равно не примут
    ][
      == Архитектура

      - Статический оптимизатор обгонит динамический
      - Отладка -- удел сильных
    ][
      == Python

      - Нет типов -- больно
      - Медленно
    ]
  )
]

// Tinygrad Capabilities
#slide[
  = Tinygrad

  #grid(
    columns: (1fr, 1fr),
    gutter: 2em,
    [
      == Аппаратные ускорители

      #list(
        [CPU],
        [CUDA],
        [AMD],
        [METAL],
        [WebGPU],
        [OpenCL],
        [Qualcomm]
      )
    ],
    [
      == Особенности

      #enum(
        [Гибкий фронтенд a la PyTorch],
        [Ленивая семантика a la MLX],
        [JIT компиляция],
        [Полу-автоматическое шардирование],
        [Богатые средства интроспекции],
        [Легко добавить новый таргет],
      )
    ]
  )
]

// Tinygrad Architecture
#slide[
  = Tinygrad: Архитектура

  #v(1em)

  #grid(
    columns: (1fr, 1fr, 1fr, 1fr, 1fr),
    gutter: 0.8em,
    align: center,
    [
      *Tensor Layer*\
      PyTorch-like API\
      Lazy evaluation\
      #sym.arrow.b\
      UOp graph
    ],
    [
      *Rangeify*\
      Eliminate movement\
      (reshape/permute)\
      #sym.arrow.b\
      Index expressions\
      \+ RANGE nodes
    ],
    [
      *Schedule*\
      Kernel extraction\
      Dependency analysis\
      Fusion\
      #sym.arrow.b\
      ScheduleItem list
    ],
    [
      *Codegen*\
      Lower UOp\
      Simplify\
      #sym.arrow.b\
      C/CUDA code
    ],                                                                                                                             [
      *Runtime*\                                                                                                                     Compile (clang/nvcc)\
      Buffer allocation\                                                                                                             #sym.arrow.b\
      Execute kernels                                                                                                              ]
  )
]

#slide[
  = Tinygrad: Что не так

  #set text(size: 20pt)

  == Python

  #list(
    [Нельзя написать нормальные биндинги в Rust],
    [Без типов программировать неприятно]
  )
]

#slide[
  = Frontend: PyTorch-like код

  #v(1em)

  == Tinygrad
  ```py
  a = Tensor([1.0, 2.0, 3.0])
  b = Tensor([4.0, 5.0, 6.0])
  c = Tensor([2.0, 2.0, 2.0])
  result = (a + b) * c
  ```

  == Morok
  ```rs
  let a = Tensor::from_slice([1.0, 2.0, 3.0]);
  let b = Tensor::from_slice([4.0, 5.0, 6.0]);
  let c = Tensor::from_slice([2.0, 2.0, 2.0]);
  let result = (a + b) * c;
  ```
]

#slide[
  = Граф вычислений

  #v(1em)

  ```rs
  pub enum Op {
      Const(ConstValueHash),
      Unique(usize),
      // ...
      Unary(UnaryOp, Rc<UOp>),
      Binary(BinaryOp, Rc<UOp>, Rc<UOp>),
      Ternary(TernaryOp, Rc<UOp>, Rc<UOp>, Rc<UOp>),
      // ...
      If { condition: Rc<UOp>, body: SmallVec<[Rc<UOp>; 4]> },
      EndIf { if_op: Rc<UOp>, body: SmallVec<[Rc<UOp>; 4]> },
  }
  ```
]

#slide[
  = Tinygrad: Спагетти вычислений

  #v(1em)

  ```py
  for u in sched_sink.toposort():
    if u.op is Ops.AFTER: continue
    k = u.src[1]  # Assumes .src has index 1!

    for s in k.src[0].src if k.op is Ops.END else k.src:  # Structure depends on op
      if s.op is Ops.AFTER:
        children[s.src[1]].append(k)
      elif s.op is Ops.BUFFER:
        pass
      else:
        # Runtime error for unhandled ops!
        raise RuntimeError(f"input must be AFTER or BUFFER, not {s.op}")
  ```
]

// Lazy Evaluation Graph
#slide[
  = Вычислительный граф

  #v(1em)

  #align(center)[
    #render(read("graph.dot"), background: rgb("#2d2d2d"))
  ]
]

#slide[
  = Оптимизатор: Graph Rewriting

  #v(1em)

  ```rs
  pub fn identity_and_zero_patterns() -> PatternMatcher {
      patterns! {
          // ========== Identity folding (commutative) ==========
          Add[x, @zero] ~> x,
          Mul[x, @one] ~> x,
          Or[x, @zero] ~> x,
          Xor[x, @zero] ~> x,
      }
  }
  ```
]

#slide[
  = Оптимизатор: Graph Rewriting
  ```rs
  /// - x + x → 2*x
  /// - (c1 * x) + (c2 * x) → (c1 + c2) * x
  /// - (x * c1) + (x * c2) → x * (c1 + c2)
  pub fn term_combining_dsl_patterns() -> PatternMatcher {
      patterns! {
          // x + x → 2*x
          Add(x, x) => 2.into_uop(x.dtype()).try_mul_op(x).ok(),
          // (c1 * x) + (c2 * x) → (c1 + c2) * x
          Add(Mul(c1 @const(c1_val), x), Mul(c2 @const(c2_val), x))
            => eval_add(c1_val, c2_val)?.into_uop(c1.dtype()).try_mul_op(x).ok(),
          // (x * c1) + (x * c2) → x * (c1 + c2)
          Add(Mul(x, c1 @const(c1_val)), Mul(x, c2 @const(c2_val)))
            => x.try_mul_op(&eval_add(c1_val, c2_val)?.into_uop(c1.dtype())).ok(),
      }
  }
  ```
]

#slide[
  = Оптимизатор: Формальная верификация

  ```rs
  #[test]
  fn test_verify_identity_add_zero() {
      // x + 0 = x
      let x = UOp::var("x", DType::Int32, 0, 100);
      let zero = UOp::const_(DType::Int32, ConstValue::Int(0));
      let x_plus_zero = UOp::try_add_op(x.clone(), zero.clone()).unwrap();

      verify_equivalence(&x_plus_zero, &x).expect("x + 0 should equal x");
  }
  ```
]

#slide[
  = Оптимизатор: Формальная верификация

  ```rs
  #[test]
  fn z3_verify_zero_mul(x in arb_var_uop(DType::Int32)) {
      let zero = UOp::const_(DType::Int32, ConstValue::Int(0));
      let expr = UOp::try_mul_op(x.clone(), zero.clone()).unwrap();
      let matcher = symbolic_simple();
      let simplified = graph_rewrite(&matcher, expr.clone());

      // Should simplify to 0
      prop_assert!(Rc::ptr_eq(&simplified, &zero));

      // Z3 should verify equivalence
      verify_equivalence(&expr, &simplified).expect("Z3 should verify x * 0 = 0");
  }
  ```
]

// Codegen Example
#slide[
  = Кодогенерация: Сгенерированный ядро

  ```c
  __kernel void E_3(
      global float* data0,           // Output: result
      const global float* data1,     // Input: a
      const global float* data2,     // Input: b
      const global float* data3)     // Input: c
  {
      int gidx0 = get_global_id(0);  // Global thread ID

      if (gidx0 < 3) {                // Bounds check
          float val0 = data1[gidx0];  // Load a[i]
          float val1 = data2[gidx0];  // Load b[i]
          float val2 = data3[gidx0];  // Load c[i]
          float acc0 = (val0 + val1);  // a + b
          float acc1 = (acc0 * val2);  // (a+b) * c
          data0[gidx0] = acc1;         // Store result
      }
  }
  ```
]

// Liho Slide
#slide[
  = История развития

  #set text(size: 20pt)

  == Эволюция портирования

  #list(
    [*Прямолинейный порт из Python* -- плохая производительность и эргономика],
    [*Подходы из взрослых компиляторов* -- излишняя сложность для задачи],
    [*Morok* -- третья итерация, баланс простоты и эффективности]
  )

  #v(1em)

  == Ключевые улучшения

  #list(
    [Переосмысление структур данных под Rust],
    [Упрощение пайплайна компиляции],
    [Использование idiomatic Rust patterns]
  )
]

// Morok Performance
#slide[
  = Morok: Производительность

  #v(1em)

  #align(center)[
    #image("performance_comparison.png", width: 95%)
  ]
]

// Thunderkittens
#slide[
  = Thunderkittens

  #set text(size: 20pt)

  == Что это?

  #list(
    [Проект *HazyResearch* (Stanford University)],
    [Минималистичный DSL для программирования GPU],
    [Высокоуровневые абстракции над CUDA]
  )

  #v(0.5em)

  == Преимущества

  #list(
    [Близко к теоретическому пределу производительности GPU],
    [Значительно проще ручного написания CUDA-ядер],
  )

  #v(0.5em)

  == Morok-kittens

  #list(
    [Схожий DSL над UOps, использующий те же принципы],
    [Flash Attention 2/3/4 на этой архитектуре]
  )
]

// Future Plans
#slide[
  = Планы

  #set text(size: 20pt)

  == Архитектура

  #list(
    [Избавиться от Cell\*],
    [Улучшить интерфейсы]
  )

  == Производительность

  #list(
    [Реализовать `morok-kittens`, портировать FA 2/3/4],
    [Оптимизировать компилятор графов]
  )

  == Экосистема

  #list(
    [Написать гайды по портированию с других фреймворков],
    [Портировать современные модели на Morok],
    [Расширить поддержку аппаратных ускорителей]
  )
]

#slide[
  #grid(
    columns: (1fr, 1fr),
    gutter: 2em,
    align: center,
    [
      == Отзывы
      #image("review.svg", width: 80%)
    ],
    [
      == GitHub
      #image("morok.svg", width: 80%)
    ]
  )
]
