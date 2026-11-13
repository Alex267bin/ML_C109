# Dataset audit and feature availability

The project uses the Corporación Favorita retail time-series dataset. The core training table contains daily sales by store and product family, including the target `sales` and promotion count `onpromotion`.

| Assignment concept | Available in Favorita data? | Project treatment |
|---|---|---|
| Product/category | Yes (`family`) | Used |
| Store/location | Yes | Used |
| Date/seasonality | Yes | Calendar features |
| Promotions | Yes (`onpromotion`) | Used |
| Holidays/events | Yes | Aggregated daily holiday indicator |
| Oil price | Yes | Used with median imputation |
| Customer demographics | No | Not fabricated |
| Reviews/ratings | No | Not fabricated |
| Customer lifetime value | No | Not fabricated |
| Advertising spend/campaign budget | No | Not fabricated |

The baseline does not use transaction totals or sales lag features. They require a clearly defined forecast-time information set to avoid target leakage.
