# Intent Personalization Validation (25 Users, Fixed Sample)

## Method
- Before artifact: `output/serving_audit/reuse_bug_after_fix_sample25_seed20260315_raw.json`
- After artifact: `output/serving_audit/intent_layer_after_fixedsample_sample25_seed20260315_raw.json`
- Both runs use the same 25 profiles (loaded directly from the before raw artifact), and the real serving path:
  `build_recommendations_for_profiles()`

## Coverage
- Before bundle count distribution: `{ "3": 25 }`
- After bundle count distribution: `{ "3": 25 }`
- Changed bundle count by user: `[]` (none)

## Top User Intent Per Sampled User (After)
- `person_3d3144ebe91a`: `staples` (0.2660)
- `person_9a4a6ed25297`: `cleaning` (0.4603)
- `person_d6c05b5e7116`: `dessert` (0.2238)
- `person_d29eb3ade2cd`: `cleaning` (0.2692)
- `person_921139a190fe`: `cleaning` (0.3550)
- `person_8031d3841609`: `dessert` (0.2289)
- `person_35cd9db7a83c`: `cleaning` (0.4351)
- `person_c6d1bc77f50f`: `staples` (0.2160)
- `person_6e42d891b9d8`: `staples` (0.3758)
- `person_f40b087c9f41`: `staples` (0.2738)
- `person_6f073e5f4c4f`: `cleaning` (0.3589)
- `person_f3d5dd7d35c4`: `preparation` (0.3424)
- `person_bce8904f8aca`: `drink_pairing` (0.2607)
- `person_7dbd4b3ab95a`: `staples` (0.2371)
- `person_7807432d75c9`: `cleaning` (0.1881)
- `person_1c792f0bee65`: `meal_base` (0.3191)
- `person_2a8cb3f67a65`: `snack` (0.2398)
- `person_51fdce51d0f4`: `meal_base` (0.2499)
- `person_e0c05c17fd30`: `staples` (0.1928)
- `person_8697381a7793`: `meal_base` (0.2651)
- `person_551f1e0907bb`: `staples` (0.3170)
- `person_e769c97fc0b1`: `dessert` (0.2592)
- `person_df253d7b4ec7`: `meal_base` (0.2398)
- `person_32f6aa7fdd0c`: `staples` (0.2064)
- `person_a0301b44bc93`: `dessert` (0.3132)

## Intent Distribution Before vs After (75 bundles)
- `cleaning`: 2 (2.67%) -> 1 (1.33%)
- `dessert`: 16 (21.33%) -> 14 (18.67%)
- `drink_pairing`: 11 (14.67%) -> 18 (24.00%)
- `meal_base`: 27 (36.00%) -> 25 (33.33%)
- `preparation`: 8 (10.67%) -> 6 (8.00%)
- `snack`: 7 (9.33%) -> 7 (9.33%)
- `staples`: 4 (5.33%) -> 4 (5.33%)

## Cleaning Bundles Served (After)
- Count: `1`
- Bundle:
  - Profile: `person_f3d5dd7d35c4`
  - Lane: `nonfood`
  - Origin: `fallback_cleaning`
  - Pair: `Shinex - Disinfectant with Haram Fragrance - 1 Liter` + `Shinex disinfectant, cleaner and deodorizer, Al Haram perfume scent, 3 liters`

## Suspicious Misclassifications
- Before: `0`
- After: `0`

## Latency Impact (25 users)
- Before:
  - total: `67.5405s`
  - avg/user: `2.7016s`
  - p50: `1.8811s`
  - p90: `2.8468s`
  - p95: `3.0957s`
- After:
  - total: `75.6219s`
  - avg/user: `3.0249s`
  - p50: `1.8433s`
  - p90: `3.3293s`
  - p95: `3.3854s`
- Delta:
  - total: `+8.0814s` (+11.96%)
  - avg/user: `+0.3233s` (+11.96%)
  - p50: `-0.0378s` (-2.01%)
  - p90: `+0.4825s` (+16.95%)
  - p95: `+0.2897s` (+9.36%)

