# Information Retrieval Assignment 1: Mandatory Benchmark Evaluation Report

**Course**: CSD358 - Information Retrieval  
**Corpus Size**: $N = 100$ clothing product descriptions  
**Search Models**: Vector Space Model (`lnc.ltc`) & Positional Inverted Index (`WITHIN/k`)  

---

## 1. Free-Text Queries (Vector Space Model - lnc.ltc Weighting)

Each query returns up to top-10 documents ranked by cosine similarity score $\frac{\vec{w_q} \cdot \vec{w_d}}{\|q\| \cdot \|d\|}$. Ties are broken by increasing docID.

### Query 1: `breathable cotton fabric`
| Rank | DocID | Category | Cosine Score | Title |
| :---: | :---: | :---: | :---: | :--- |
| 1 | **D014** | Kurta | `0.2413` | Women's Printed Straight Kurta - Beige |
| 2 | **D074** | Kurta | `0.2413` | Women's Printed Straight Kurta - Pink |
| 3 | **D001** | T-Shirt | `0.2384` | Men's Cotton Crew Neck T-Shirt - Black |
| 4 | **D041** | T-Shirt | `0.2384` | Men's Cotton Crew Neck T-Shirt - Maroon |
| 5 | **D061** | T-Shirt | `0.2384` | Men's Cotton Crew Neck T-Shirt - White |
| 6 | **D081** | T-Shirt | `0.2361` | Men's Cotton Crew Neck T-Shirt - Mustard |
| 7 | **D021** | T-Shirt | `0.2312` | Men's Cotton Crew Neck T-Shirt - Sky Blue |
| 8 | **D034** | Kurta | `0.2288` | Women's Printed Straight Kurta - Maroon |
| 9 | **D094** | Kurta | `0.2288` | Women's Printed Straight Kurta - Teal |
| 10 | **D054** | Kurta | `0.2268` | Women's Printed Straight Kurta - Mustard |

### Query 2: `casual winter wear`
| Rank | DocID | Category | Cosine Score | Title |
| :---: | :---: | :---: | :---: | :--- |
| 1 | **D058** | Jacket | `0.2242` | Women's Quilted Winter Jacket - Black |
| 2 | **D018** | Jacket | `0.2227` | Women's Quilted Winter Jacket - Beige |
| 3 | **D078** | Jacket | `0.2227` | Women's Quilted Winter Jacket - Beige |
| 4 | **D038** | Jacket | `0.2202` | Women's Quilted Winter Jacket - Olive |
| 5 | **D098** | Jacket | `0.2202` | Women's Quilted Winter Jacket - Olive |
| 6 | **D010** | Sweatshirt | `0.1504` | Women's Fleece Sweatshirt - Black |
| 7 | **D070** | Sweatshirt | `0.1504` | Women's Fleece Sweatshirt - Black |
| 8 | **D028** | Jacket | `0.1501` | Women's Casual Puffer Jacket - Black |
| 9 | **D088** | Jacket | `0.1501` | Women's Casual Puffer Jacket - Black |
| 10 | **D030** | Sweatshirt | `0.1494` | Women's Fleece Sweatshirt - Maroon |

### Query 3: `regular fit cotton shirt`
| Rank | DocID | Category | Cosine Score | Title |
| :---: | :---: | :---: | :---: | :--- |
| 1 | **D022** | Shirt | `0.3044` | Men's Checked Cotton Shirt - White |
| 2 | **D082** | Shirt | `0.3044` | Men's Checked Cotton Shirt - Blue |
| 3 | **D042** | Shirt | `0.3024` | Men's Checked Cotton Shirt - Maroon |
| 4 | **D002** | Shirt | `0.2992` | Men's Checked Cotton Shirt - Black |
| 5 | **D062** | Shirt | `0.2992` | Men's Checked Cotton Shirt - Olive |
| 6 | **D052** | Shirt | `0.2502` | Women's Striped Casual Shirt - Beige |
| 7 | **D012** | Shirt | `0.2486` | Women's Striped Casual Shirt - Blue |
| 8 | **D072** | Shirt | `0.2486` | Women's Striped Casual Shirt - Black |
| 9 | **D092** | Shirt | `0.2459` | Women's Striped Casual Shirt - White |
| 10 | **D032** | Shirt | `0.2404` | Women's Striped Casual Shirt - Light Pink |

### Query 4: `comfort stretch denim jeans`
| Rank | DocID | Category | Cosine Score | Title |
| :---: | :---: | :---: | :---: | :--- |
| 1 | **D013** | Jeans | `0.3319` | Men's Slim Fit Stretch Jeans - Grey |
| 2 | **D073** | Jeans | `0.3319` | Men's Slim Fit Stretch Jeans - Grey |
| 3 | **D043** | Jeans | `0.3309` | Men's Regular Fit Denim Jeans - Grey |
| 4 | **D033** | Jeans | `0.3262` | Men's Slim Fit Stretch Jeans - Grey |
| 5 | **D093** | Jeans | `0.3262` | Men's Slim Fit Stretch Jeans - Grey |
| 6 | **D003** | Jeans | `0.3252` | Men's Regular Fit Denim Jeans - Grey |
| 7 | **D063** | Jeans | `0.3252` | Men's Regular Fit Denim Jeans - Grey |
| 8 | **D053** | Jeans | `0.3179` | Men's Slim Fit Stretch Jeans - Grey |
| 9 | **D023** | Jeans | `0.2507` | Men's Regular Fit Denim Jeans - Grey |
| 10 | **D083** | Jeans | `0.2507` | Men's Regular Fit Denim Jeans - Grey |

### Query 5: `printed daily wear saree`
| Rank | DocID | Category | Cosine Score | Title |
| :---: | :---: | :---: | :---: | :--- |
| 1 | **D005** | Saree | `0.3490` | Women's Printed Daily Wear Saree - Blue |
| 2 | **D025** | Saree | `0.3490` | Women's Printed Daily Wear Saree - Maroon |
| 3 | **D065** | Saree | `0.3490` | Women's Printed Daily Wear Saree - Green |
| 4 | **D085** | Saree | `0.3490` | Women's Printed Daily Wear Saree - Red |
| 5 | **D045** | Saree | `0.3452` | Women's Printed Daily Wear Saree - Purple |
| 6 | **D035** | Saree | `0.1814` | Women's Cotton Handloom Saree - Yellow |
| 7 | **D055** | Saree | `0.1814` | Women's Cotton Handloom Saree - Pink |
| 8 | **D095** | Saree | `0.1814` | Women's Cotton Handloom Saree - Maroon |
| 9 | **D015** | Saree | `0.1795` | Women's Cotton Handloom Saree - Red |
| 10 | **D075** | Saree | `0.1795` | Women's Cotton Handloom Saree - Blue |

### Query 6: `fleece pullover hoodie`
| Rank | DocID | Category | Cosine Score | Title |
| :---: | :---: | :---: | :---: | :--- |
| 1 | **D037** | Hoodie | `0.3549` | Unisex Fleece Pullover Hoodie - Black |
| 2 | **D097** | Hoodie | `0.3549` | Unisex Fleece Pullover Hoodie - Black |
| 3 | **D057** | Hoodie | `0.3511` | Unisex Fleece Pullover Hoodie - Wine |
| 4 | **D017** | Hoodie | `0.3467` | Unisex Fleece Pullover Hoodie - Navy Blue |
| 5 | **D077** | Hoodie | `0.3467` | Unisex Fleece Pullover Hoodie - Navy Blue |
| 6 | **D007** | Hoodie | `0.2029` | Women's Oversized Fleece Hoodie - Black |
| 7 | **D067** | Hoodie | `0.2029` | Women's Oversized Fleece Hoodie - Black |
| 8 | **D027** | Hoodie | `0.2007` | Women's Oversized Fleece Hoodie - Wine |
| 9 | **D087** | Hoodie | `0.2007` | Women's Oversized Fleece Hoodie - Wine |
| 10 | **D047** | Hoodie | `0.1983` | Women's Oversized Fleece Hoodie - Navy Blue |

### Query 7: `quilted winter jacket`
| Rank | DocID | Category | Cosine Score | Title |
| :---: | :---: | :---: | :---: | :--- |
| 1 | **D058** | Jacket | `0.3667` | Women's Quilted Winter Jacket - Black |
| 2 | **D018** | Jacket | `0.3641` | Women's Quilted Winter Jacket - Beige |
| 3 | **D078** | Jacket | `0.3641` | Women's Quilted Winter Jacket - Beige |
| 4 | **D038** | Jacket | `0.3601` | Women's Quilted Winter Jacket - Olive |
| 5 | **D098** | Jacket | `0.3601` | Women's Quilted Winter Jacket - Olive |
| 6 | **D028** | Jacket | `0.3022` | Women's Casual Puffer Jacket - Black |
| 7 | **D088** | Jacket | `0.3022` | Women's Casual Puffer Jacket - Black |
| 8 | **D048** | Jacket | `0.3001` | Women's Casual Puffer Jacket - Beige |
| 9 | **D008** | Jacket | `0.2968` | Women's Casual Puffer Jacket - Olive |
| 10 | **D068** | Jacket | `0.2968` | Women's Casual Puffer Jacket - Olive |

### Query 8: `high waist stretch leggings`
| Rank | DocID | Category | Cosine Score | Title |
| :---: | :---: | :---: | :---: | :--- |
| 1 | **D029** | Leggings | `0.4318` | Women's High Waist Stretch Leggings - Grey |
| 2 | **D049** | Leggings | `0.4318` | Women's High Waist Stretch Leggings - Black |
| 3 | **D089** | Leggings | `0.4318` | Women's High Waist Stretch Leggings - Grey |
| 4 | **D009** | Leggings | `0.4178` | Women's High Waist Stretch Leggings - Olive Green |
| 5 | **D069** | Leggings | `0.4178` | Women's High Waist Stretch Leggings - Olive Green |
| 6 | **D019** | Leggings | `0.2620` | Women's Printed Active Leggings - Black |
| 7 | **D059** | Leggings | `0.2620` | Women's Printed Active Leggings - Grey |
| 8 | **D079** | Leggings | `0.2620` | Women's Printed Active Leggings - Black |
| 9 | **D039** | Leggings | `0.2535` | Women's Printed Active Leggings - Olive Green |
| 10 | **D099** | Leggings | `0.2535` | Women's Printed Active Leggings - Olive Green |

### Query 9: `warm fleece sweatshirt`
| Rank | DocID | Category | Cosine Score | Title |
| :---: | :---: | :---: | :---: | :--- |
| 1 | **D010** | Sweatshirt | `0.3318` | Women's Fleece Sweatshirt - Black |
| 2 | **D070** | Sweatshirt | `0.3318` | Women's Fleece Sweatshirt - Black |
| 3 | **D030** | Sweatshirt | `0.3295` | Women's Fleece Sweatshirt - Maroon |
| 4 | **D090** | Sweatshirt | `0.3295` | Women's Fleece Sweatshirt - Maroon |
| 5 | **D050** | Sweatshirt | `0.3194` | Women's Fleece Sweatshirt - Navy Blue |
| 6 | **D040** | Sweatshirt | `0.2194` | Men's Regular Fit Sweatshirt - Black |
| 7 | **D100** | Sweatshirt | `0.2194` | Men's Regular Fit Sweatshirt - Black |
| 8 | **D060** | Sweatshirt | `0.2179` | Men's Regular Fit Sweatshirt - Maroon |
| 9 | **D020** | Sweatshirt | `0.2114` | Men's Regular Fit Sweatshirt - Navy Blue |
| 10 | **D080** | Sweatshirt | `0.2114` | Men's Regular Fit Sweatshirt - Navy Blue |

### Query 10: `embroidered festive kurta`
| Rank | DocID | Category | Cosine Score | Title |
| :---: | :---: | :---: | :---: | :--- |
| 1 | **D004** | Kurta | `0.2210` | Men's Regular Fit Kurta - Pink |
| 2 | **D064** | Kurta | `0.2210` | Men's Regular Fit Kurta - White |
| 3 | **D024** | Kurta | `0.2190` | Men's Regular Fit Kurta - Teal |
| 4 | **D084** | Kurta | `0.2190` | Men's Regular Fit Kurta - Beige |
| 5 | **D034** | Kurta | `0.2185` | Women's Printed Straight Kurta - Maroon |
| 6 | **D094** | Kurta | `0.2185` | Women's Printed Straight Kurta - Teal |
| 7 | **D054** | Kurta | `0.2166` | Women's Printed Straight Kurta - Mustard |
| 8 | **D014** | Kurta | `0.2150` | Women's Printed Straight Kurta - Beige |
| 9 | **D074** | Kurta | `0.2150` | Women's Printed Straight Kurta - Pink |
| 10 | **D044** | Kurta | `0.2124` | Men's Regular Fit Kurta - Navy Blue |

---
## 2. Exact Phrase Queries (Positional Index Intersect)

Matches documents where query terms occur consecutively in exact order: $pos_{i+1} = pos_i + 1$.

### Phrase Query 1: `"cotton shirt"`
*Total matching documents: 5*
| DocID | Category | Matching Token Positions | Title |
| :---: | :---: | :--- | :--- |
| **D002** | Shirt | `(3, 4), (8, 9)` | Men's Checked Cotton Shirt - Black |
| **D022** | Shirt | `(3, 4), (8, 9)` | Men's Checked Cotton Shirt - White |
| **D042** | Shirt | `(3, 4), (8, 9)` | Men's Checked Cotton Shirt - Maroon |
| **D062** | Shirt | `(3, 4), (8, 9)` | Men's Checked Cotton Shirt - Olive |
| **D082** | Shirt | `(3, 4), (8, 9)` | Men's Checked Cotton Shirt - Blue |

### Phrase Query 2: `"stretch denim"`
*Total matching documents: 7*
| DocID | Category | Matching Token Positions | Title |
| :---: | :---: | :--- | :--- |
| **D003** | Jeans | `(15, 16)` | Men's Regular Fit Denim Jeans - Grey |
| **D013** | Jeans | `(14, 15)` | Men's Slim Fit Stretch Jeans - Grey |
| **D033** | Jeans | `(15, 16)` | Men's Slim Fit Stretch Jeans - Grey |
| **D043** | Jeans | `(14, 15)` | Men's Regular Fit Denim Jeans - Grey |
| **D063** | Jeans | `(15, 16)` | Men's Regular Fit Denim Jeans - Grey |
| **D073** | Jeans | `(14, 15)` | Men's Slim Fit Stretch Jeans - Grey |
| **D093** | Jeans | `(15, 16)` | Men's Slim Fit Stretch Jeans - Grey |

### Phrase Query 3: `"festive wear"`
*Total matching documents: 10*
| DocID | Category | Matching Token Positions | Title |
| :---: | :---: | :--- | :--- |
| **D005** | Saree | `(21, 22)` | Women's Printed Daily Wear Saree - Blue |
| **D015** | Saree | `(19, 20)` | Women's Cotton Handloom Saree - Red |
| **D025** | Saree | `(21, 22)` | Women's Printed Daily Wear Saree - Maroon |
| **D035** | Saree | `(19, 20)` | Women's Cotton Handloom Saree - Yellow |
| **D045** | Saree | `(21, 22)` | Women's Printed Daily Wear Saree - Purple |
| **D055** | Saree | `(19, 20)` | Women's Cotton Handloom Saree - Pink |
| **D065** | Saree | `(21, 22)` | Women's Printed Daily Wear Saree - Green |
| **D075** | Saree | `(19, 20)` | Women's Cotton Handloom Saree - Blue |
| **D085** | Saree | `(21, 22)` | Women's Printed Daily Wear Saree - Red |
| **D095** | Saree | `(19, 20)` | Women's Cotton Handloom Saree - Maroon |

### Phrase Query 4: `"winter wear"`
*Total matching documents: 20*
| DocID | Category | Matching Token Positions | Title |
| :---: | :---: | :--- | :--- |
| **D008** | Jacket | `(21, 22)` | Women's Casual Puffer Jacket - Olive |
| **D010** | Sweatshirt | `(18, 19)` | Women's Fleece Sweatshirt - Black |
| **D018** | Jacket | `(20, 21)` | Women's Quilted Winter Jacket - Beige |
| **D020** | Sweatshirt | `(22, 23)` | Men's Regular Fit Sweatshirt - Navy Blue |
| **D028** | Jacket | `(21, 22)` | Women's Casual Puffer Jacket - Black |
| **D030** | Sweatshirt | `(18, 19)` | Women's Fleece Sweatshirt - Maroon |
| **D038** | Jacket | `(20, 21)` | Women's Quilted Winter Jacket - Olive |
| **D040** | Sweatshirt | `(20, 21)` | Men's Regular Fit Sweatshirt - Black |
| **D048** | Jacket | `(21, 22)` | Women's Casual Puffer Jacket - Beige |
| **D050** | Sweatshirt | `(20, 21)` | Women's Fleece Sweatshirt - Navy Blue |

### Phrase Query 5: `"regular fit"`
*Total matching documents: 15*
| DocID | Category | Matching Token Positions | Title |
| :---: | :---: | :--- | :--- |
| **D003** | Jeans | `(2, 3), (8, 9)` | Men's Regular Fit Denim Jeans - Grey |
| **D004** | Kurta | `(2, 3), (7, 8)` | Men's Regular Fit Kurta - Pink |
| **D020** | Sweatshirt | `(2, 3), (8, 9)` | Men's Regular Fit Sweatshirt - Navy Blue |
| **D023** | Jeans | `(2, 3), (8, 9)` | Men's Regular Fit Denim Jeans - Grey |
| **D024** | Kurta | `(2, 3), (7, 8)` | Men's Regular Fit Kurta - Teal |
| **D040** | Sweatshirt | `(2, 3), (7, 8)` | Men's Regular Fit Sweatshirt - Black |
| **D043** | Jeans | `(2, 3), (8, 9)` | Men's Regular Fit Denim Jeans - Grey |
| **D044** | Kurta | `(2, 3), (8, 9)` | Men's Regular Fit Kurta - Navy Blue |
| **D060** | Sweatshirt | `(2, 3), (7, 8)` | Men's Regular Fit Sweatshirt - Maroon |
| **D063** | Jeans | `(2, 3), (8, 9)` | Men's Regular Fit Denim Jeans - Grey |

---
## 3. Ordered Proximity Queries (`WITHIN/k`)

Matches documents where term2 occurs *after* term1 within at most $k$ tokens: $1 \le pos_2 - pos_1 \le k$.

### Proximity Query 1: `cotton WITHIN/3 shirt` (k = 3)
*Total matching documents: 15*
| DocID | Category | Matching Position Pairs (pos1, pos2) | Title |
| :---: | :---: | :--- | :--- |
| **D001** | T-Shirt | `(17, 19)` | Men's Cotton Crew Neck T-Shirt - Black |
| **D002** | Shirt | `(3, 4), (8, 9)` | Men's Checked Cotton Shirt - Black |
| **D011** | T-Shirt | `(14, 17)` | Men's Oversized Graphic T-Shirt - Mustard |
| **D021** | T-Shirt | `(19, 21)` | Men's Cotton Crew Neck T-Shirt - Sky Blue |
| **D022** | Shirt | `(3, 4), (8, 9)` | Men's Checked Cotton Shirt - White |
| **D031** | T-Shirt | `(16, 19)` | Men's Oversized Graphic T-Shirt - Olive Green |
| **D041** | T-Shirt | `(17, 19)` | Men's Cotton Crew Neck T-Shirt - Maroon |
| **D042** | Shirt | `(3, 4), (8, 9)` | Men's Checked Cotton Shirt - Maroon |
| **D051** | T-Shirt | `(16, 19)` | Men's Oversized Graphic T-Shirt - Navy Blue |
| **D061** | T-Shirt | `(17, 19)` | Men's Cotton Crew Neck T-Shirt - White |

### Proximity Query 2: `stretch WITHIN/4 denim` (k = 4)
*Total matching documents: 7*
| DocID | Category | Matching Position Pairs (pos1, pos2) | Title |
| :---: | :---: | :--- | :--- |
| **D003** | Jeans | `(15, 16)` | Men's Regular Fit Denim Jeans - Grey |
| **D013** | Jeans | `(14, 15)` | Men's Slim Fit Stretch Jeans - Grey |
| **D033** | Jeans | `(15, 16)` | Men's Slim Fit Stretch Jeans - Grey |
| **D043** | Jeans | `(14, 15)` | Men's Regular Fit Denim Jeans - Grey |
| **D063** | Jeans | `(15, 16)` | Men's Regular Fit Denim Jeans - Grey |
| **D073** | Jeans | `(14, 15)` | Men's Slim Fit Stretch Jeans - Grey |
| **D093** | Jeans | `(15, 16)` | Men's Slim Fit Stretch Jeans - Grey |

### Proximity Query 3: `winter WITHIN/3 wear` (k = 3)
*Total matching documents: 20*
| DocID | Category | Matching Position Pairs (pos1, pos2) | Title |
| :---: | :---: | :--- | :--- |
| **D008** | Jacket | `(21, 22)` | Women's Casual Puffer Jacket - Olive |
| **D010** | Sweatshirt | `(18, 19)` | Women's Fleece Sweatshirt - Black |
| **D018** | Jacket | `(20, 21)` | Women's Quilted Winter Jacket - Beige |
| **D020** | Sweatshirt | `(22, 23)` | Men's Regular Fit Sweatshirt - Navy Blue |
| **D028** | Jacket | `(21, 22)` | Women's Casual Puffer Jacket - Black |
| **D030** | Sweatshirt | `(18, 19)` | Women's Fleece Sweatshirt - Maroon |
| **D038** | Jacket | `(20, 21)` | Women's Quilted Winter Jacket - Olive |
| **D040** | Sweatshirt | `(20, 21)` | Men's Regular Fit Sweatshirt - Black |
| **D048** | Jacket | `(21, 22)` | Women's Casual Puffer Jacket - Beige |
| **D050** | Sweatshirt | `(20, 21)` | Women's Fleece Sweatshirt - Navy Blue |

### Proximity Query 4: `festive WITHIN/4 kurta` (k = 4)
*No documents matched proximity condition.*

---
## 4. Non-Corpus / Out-of-Vocabulary Query

### Query: `waterproof spacesuit astronaut`
- **VSM Matches**: 0 documents returned.
- **Positional Matches**: 0 documents returned.
- **Behavior Explanation**: Since terms `waterproof`, `spacesuit`, and `astronaut` do not exist in the corpus dictionary ($df = 0$), $w_{q,t} = 0$, producing a null query vector $\|q\| = 0$. The system correctly and gracefully returns an empty result set without errors.

---
## 5. Comparative Analysis: VSM vs. Positional Retrieval

The assignment requires explaining at least **two concrete cases where positional information changes the result set/order**.

### Case Study 1: Query `"cotton shirt"`
- **User Intent**: The user is looking specifically for a cotton dress shirt or casual button-down shirt.
- **VSM (lnc.ltc) Result**: Returns 10 documents. VSM treats the query as a bag-of-words. As a result, T-Shirt documents (e.g. `D001`, `D011`, `D021`, `D041`) rank highly because they contain `cotton` in high frequency alongside `t-shirt` (whose stemmed form conflates with `shirt`).
- **Positional Index Result**: Strictly returns 5 documents (`D002`, `D022`, `D042`, `D062`, `D082`) where `cotton` is immediately adjacent to `shirt` ($pos_2 = pos_1 + 1$).
- **Impact of Positional Information**: Eliminates **false positives** where the words appeared disjointly or in reverse/separated order. It turns an ambiguous broad match into high-precision shirt retrieval.

### Case Study 2: Query `"stretch denim"`
- **User Intent**: The user wants clothing items woven specifically with elastic stretch denim fabric.
- **VSM Result**: Returns 10 documents. Documents containing `stretch` (like leggings `D009`, `D029`) or `denim` (like jackets `D018`, `D038`) get accumulated dot-product scores even though neither is made of actual 'stretch denim'.
- **Positional Index Result**: Returns 7 jeans documents (`D003`, `D013`, `D033`, `D043`, `D063`, `D073`, `D093`) where `stretch` and `denim` appear consecutively at positions `(14, 15)` or `(15, 16)`.
- **Impact of Positional Information**: Positional constraint enforces multi-word semantic binding, completely filtering out unrelated denim jackets and spandex leggings that share individual terms.

### Case Study 3: Proximity vs. Exact Phrase (`cotton WITHIN/3 shirt`)
- **Exact Phrase (`"cotton shirt"`)**: Matches 5 documents where tokens are strictly adjacent ($pos_2 = pos_1 + 1$).
- **Proximity Query (`cotton WITHIN/3 shirt`)**: Matches 15 documents. By relaxing the distance window to $k=3$, it successfully retrieves both strict shirts (positions `(3, 4)`) AND cotton crew-neck t-shirts (positions `(17, 19)`, where 'crew' and 'neck' intervene).
- **Insight**: Proximity queries provide a tunable sweet spot between the ultra-rigid precision of phrase search and the noisy bag-of-words recall of VSM.
