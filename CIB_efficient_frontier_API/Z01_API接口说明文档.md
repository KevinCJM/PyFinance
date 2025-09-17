# **投资组合分析计算 API 文档**

## **1. 概述**

### **1.1. 目标**

本系列 API 提供了一套完整的投资组合分析工具，旨在为前端应用提供强大的后端计算支持。其核心功能包括：
1.  **计算可配置空间与有效前沿**: 根据不同约束（市场、标准组合、客户持仓），生成大量随机投资组合，并找出其中的有效前沿。
2.  **计算风险边界**: 在给定的资产权重约束下，精确寻找风险最高和最低的两个极值组合。
3.  **寻找理想推荐组合**: 基于客户的现有持仓和一条已知的有效前沿，推荐四个优化目标点（同收益风险最小、同风险收益最大、换仓最小、综合更优）。

### **1.2. 通用说明**

*   **方法**: 所有接口均使用 `POST` 方法。
*   **请求头**: `Content-Type: application/json`
*   **数据文件**: 所有计算都依赖于服务器端的 `历史净值数据_万得指数.xlsx` 文件。调用方无需关心此文件。

---

## **2. 通用响应格式**

### **2.1. 成功响应**

当所有请求的计算都成功完成时，HTTP状态码为 `200 OK`，响应体结构如下：

```json
{
  "success": true,
  "data": { ... }
}
```
*   `success`: (布尔型) 始终为 `true`。
*   `data`: (对象) 包含该接口返回的具体业务数据。

### **2.2. 失败响应**

当请求处理过程中发生任何错误时，HTTP状态码可能为 `400` (客户端错误) 或 `500` (服务器内部错误)，响应体结构如下：

```json
{
  "success": false,
  "error_code": "ERROR_CODE_STRING",
  "message": "详细的错误描述信息"
}
```
*   `success`: (布尔型) 始终为 `false`。
*   `error_code`: (字符串) 错误的唯一标识码，便于程序化处理。
*   `message`: (字符串) 对错误的详细文字说明。

---

## **3. 功能一: 计算可配置空间与有效前沿 (`A01_main_api.py`)**

*   **接口路径**: `/api/v1/calculate-portfolios` (暂定)
*   **功能描述**: 此接口是核心功能，能够根据不同的约束条件，计算并返回投资组合的**可配置空间**（大量满足约束的随机权重组合）与**有效前沿**（在给定风险下收益最优的组合边界）。它支持在单次请求中组合调用以下三种模块。

### **3.1. 模块 1: 计算无约束的市场组合**

*   **触发条件**: 请求体中包含 `"cal_market_ef": true`。
*   **请求参数 (`入参`)**: 
    *   `asset_list` (Array[String], **必需**): 定义计算所涉及的资产名称列表。后续所有权重数组的顺序都与此对应。
    *   `cal_market_ef` (Boolean, **必需**): 必须设置为 `true` 来触发此功能的计算。
*   **请求示例**:
    ```json
    {
      "asset_list": ["货币现金类", "固定收益类", "权益投资类"],
      "cal_market_ef": true
    }
    ```

*   **响应结果 (`出参`)**: 
    *   成功时，响应的 `data` 对象中将包含一个 `market` 键，其值的结构如下：
        *   `weights` (Array[Array[Float]]): 二维数组，代表所有生成的投资组合权重。
        *   `ret_annual` (Array[Float]): 一维数组，包含每个组合对应的**年化收益率**。
        *   `risk_arr` (Array[Float]): 一维数组，包含每个组合对应的**年化风险值**。
        *   `ef_mask` (Array[Boolean]): 一维布尔数组，`true` 表示对应索引的组合位于有效前沿上。
    *   **响应示例**:
    ```json
    {
        "success": true,
        "data": {
            "market": {
                "weights": [
                    [0.1, 0.2, 0.7], 
                    [0.2, 0.3, 0.5], 
                    ...
                ],
                "ret_annual": [0.05, 0.06, ...],
                "risk_arr": [0.08, 0.09, ...],
                "ef_mask": [false, true, ...]
            }
        }
    }
    ```

### **3.2. 模块 2: 计算标准组合 (C1-C6)**

*   **触发条件**: 请求体中包含 `WeightRange` 和 `StandardProportion` 对象。
*   **请求参数 (`入参`)**: 
    *   `asset_list` (Array[String], **必需**): 资产名称列表。
    *   `WeightRange` (Object, **必需**): 定义**每个标准组合**的大类资产权重上下限。外层Key为组合名称（如"C1"），内层Key为资产名称，Value为 `[最小权重, 最大权重]`。
    *   `StandardProportion` (Object, **必需**): 定义**每个标准组合**的官方标准权重配比。外层Key为组合名称，内层Key为资产名称，Value为权重比例。
    *   **请求示例**:
    ```json
    {
      "asset_list": ["货币现金类", "固定收益类", "权益投资类"],
      "WeightRange": {
        "C4": {
          "货币现金类": [0.0, 1.0],
          "固定收益类": [0.0, 1.0],
          "权益投资类": [0.0, 0.2]
        },
        "C5": {
          "货币现金类": [0.0, 1.0],
          "固定收益类": [0.0, 1.0],
          "权益投资类": [0.0, 0.5]
        }
      },
      "StandardProportion": {
        "C4": {
          "货币现金类": 0.05,
          "固定收益类": 0.4,
          "权益投资类": 0.55
        },
        "C5": {
          "货币现金类": 0.05,
          "固定收益类": 0.2,
          "权益投资类": 0.75
        }
      }
    }
    ```

*   **响应结果 (`出参`)**: 
    *   成功时，响应的 `data` 对象中将包含一个 `standard` 键。它是一个以标准组合名称（C1, C2, ...）为Key的对象，每个Key的值都遵循 **3.1** 中 `market` 的结果结构。
    *   **响应示例**:
    ```json
    {
        "success": true,
        "data": {
            "standard": {
                "C4": {
                    "weights": [...],
                    "ret_annual": [...],
                    "risk_arr": [...],
                    "ef_mask": [...]
                },
                "C5": {
                    "weights": [...],
                    "ret_annual": [...],
                    "risk_arr": [...],
                    "ef_mask": [...]
                }
            }
        }
    }
    ```

### **3.3. 模块 3: 计算客户持仓组合**

*   **触发条件**: 请求体中包含 `user_holding` 对象。
*   **请求参数 (`入参`)**: 
    *   `asset_list` (Array[String], **必需**): 资产名称列表。
    *   `user_holding` (Object, **必需**): 封装所有与客户相关的约束信息。
        *   `WeightRange` (Object): 客户风险等级(C1~C6)对应的权重上下限。 `资产名称: [最小权重, 最大权重]`。
        *   `StandardProportion` (Object): 客户**当前的实际持仓比例**。 `资产名称: 权重`。
        *   `can_sell` (Object): 各大类资产**是否允许卖出**的布尔标记。`true`表示允许卖出（权重可低于当前值），`false`表示不允许（权重须不低于当前值）。
        *   `can_buy` (Object): 各大类资产**是否允许买入**的布尔标记。`true`表示允许买入（权重可高于当前值），`false`表示不允许（权重须不高于当前值）。
    *   **请求示例**:
    ```json
    {
      "asset_list": ["货币现金类", "固定收益类", "权益投资类"],
      "user_holding": {
        "WeightRange": {
          "货币现金类": [0.0, 1.0],
          "固定收益类": [0.0, 1.0],
          "权益投资类": [0.0, 0.7]
        },
        "StandardProportion": {
          "货币现金类": 0.1,
          "固定收益类": 0.5,
          "权益投资类": 0.4
        },
        "can_sell": {
          "货币现金类": false,
          "固定收益类": true,
          "权益投资类": true
        },
        "can_buy": {
          "货币现金类": true,
          "固定收益类": true,
          "权益投资类": false
        }
      }
    }
    ```

*   **响应结果 (`出参`)**: 
    *   成功时，响应的 `data` 对象中将包含一个 `user` 键，其值的结构与 **3.1** 中 `market` 键的结构完全相同，包含了客户专属的 `weights`, `ret_annual`, `risk_arr`, `ef_mask`。
    *   **响应示例**:
    ```json
    {
        "success": true,
        "data": {
            "user": {
                "weights": [...],
                "ret_annual": [...],
                "risk_arr": [...],
                "ef_mask": [...]
            }
        }
    }
    ```

---

## **4. 功能二: 计算风险边界组合 (`A02_risk_boundaries_api.py`)**

*   **接口路径**: `/api/v1/risk-boundaries` (暂定)
*   **功能描述**: 在给定的大类资产权重约束下，通过优化算法精确寻找出**风险最低**和**风险最高**的两个极值投资组合。

*   **请求参数 (`入参`)**:
    ```json
    {
      "asset_list": [ "货币现金类", "固定收益类", "混合策略类", "权益投资类", "另类投资类" ],
      "WeightRange": {
        "货币现金类": [0.04, 0.06], 
        "固定收益类": [0.08, 0.12],
        "混合策略类": [0.12, 0.18], 
        "权益投资类": [0.48, 0.72],
        "另类投资类": [0.08, 0.12]
      }
    }
    ```
    *   `asset_list` (Array[String], **必需**): 资产名称列表。
    *   `WeightRange` (Object, **必需**): 定义每个大类资产的权重**上下限**。Key为资产名称，Value为 `[最小权重, 最大权重]`。

*   **响应结果 (`出参`)**:
    ```json
    {
        "success": true,
        "data": {
            "min_risk": {
                "risk_value": 0.0856,
                "weights": [0.06, 0.12, 0.18, 0.48, 0.16]
            },
            "max_risk": {
                "risk_value": 0.1523,
                "weights": [0.04, 0.08, 0.12, 0.72, 0.04]
            }
        }
    }
    ```
    *   `min_risk` (Object): 最小风险组合。
        *   `risk_value` (Float): 年化风险值。
        *   `weights` (Array[Float]): 对应的资产权重数组。
    *   `max_risk` (Object): 最大风险组合。
        *   `risk_value` (Float): 年化风险值。
        *   `weights` (Array[Float]): 对应的资产权重数组。

---

## **5. 功能三: 寻找理想推荐组合 (`A03_ideal_portfolio_api.py`)**

*   **接口路径**: `/api/v1/ideal-portfolios` (暂定)
*   **功能描述**: 接收客户的**当前持仓**以及一条**已计算好的有效前沿**（通常来自功能一），然后基于这些信息，为客户寻找并推荐四个具有明确优化目标的“理想组合”。

*   **请求参数 (`入参`)**:
    *   `asset_list` (Array[String], **必需**): 资产名称列表。
    *   `user_holding` (Object, **必需**): 客户持仓信息，结构与 **3.3** 中的 `user_holding` 完全相同。
    *   `ef_data` (Object, **必需**): **有效前沿数据**。这是本接口的关键输入，必须包含由 `功能一` 计算出的有效前沿部分的权重。
        *   `weights` (Array[Array[Float]]): 有效前沿组合的权重二维数组。
    *   `refine_ef_before_select` (Boolean, 可选, 默认`false`): 是否在选择“换仓最小”点之前对有效前沿进行精炼，以获得更优结果。建议保持默认值。

*   **请求示例**:
    ```json
    {
        "asset_list": ["货币现金类", "固定收益类", "混合策略类", "权益投资类", "另类投资类"],
        "user_holding": {
            "WeightRange": { "货币现金类": [0.0, 1.0], ... },
            "StandardProportion": { "货币现金类": 0.1, ... },
            "can_sell": { "货币现金类": true, ... },
            "can_buy": { "货币现金类": true, ... }
        },
        "ef_data": {
            "weights": [ 
                [0.1, 0.2, 0.3, 0.2, 0.2],
                [0.1, 0.2, 0.2, 0.3, 0.2],
                ... 
            ]
        }
    }
    ```

*   **响应结果 (`出参`)**:
    *   `data` 对象包含四个推荐组合，每个组合都包含 `weights`, `ret_annual`, `risk_value`, `turnover_l1_half` (单边换手率) 字段。
    ```json
    {
        "success": true,
        "data": {
            "same_return_min_risk": {
                "weights": [...], "ret_annual": 0.05, "risk_value": 0.07, "turnover_l1_half": 0.15
            },
            "same_risk_max_return": {
                "weights": [...], "ret_annual": 0.06, "risk_value": 0.08, "turnover_l1_half": 0.18
            },
            "min_turnover_on_ef": {
                "weights": [...], "ret_annual": 0.055, "risk_value": 0.075, "turnover_l1_half": 0.10
            },
            "better_return_lower_risk": {
                "weights": [...], "ret_annual": 0.058, "risk_value": 0.072, "turnover_l1_half": 0.16
            }
        }
    }
    ```
    *   `same_return_min_risk`: **同等收益，风险最小**的组合。
    *   `same_risk_max_return`: **同等风险，收益最高**的组合。
    *   `min_turnover_on_ef`: 在有效前沿上**换手率最小**的组合。
    *   `better_return_lower_risk`: 一个综合更优的组合，寻求**更高收益**和**更低风险**的平衡点。

---

## **6. 错误码说明**

| Error Code                 | 含义                       | 建议处理方式                                         |
| -------------------------- | -------------------------- | ---------------------------------------------------- |
| `INVALID_JSON_INPUT`       | 请求体不是一个合法的JSON   | 检查请求体JSON语法是否正确。                         |
| `MISSING_OR_INVALID_FIELD` | 缺少必需字段或字段类型错误 | 对照文档检查 `asset_list` 等必需字段是否存在且格式正确。 |
| `DATA_FILE_NOT_FOUND`      | 服务器端依赖的数据文件缺失 | 这是一个服务器端问题，请联系Python服务维护人员。     |
| `INVALID_DATA_OR_CONFIG`   | 输入的业务数据或配置不合法 | 检查传入的约束值是否有效（例如，权重范围是否合理）。 |
| `INTERNAL_SERVER_ERROR`    | 服务器内部未知计算错误     | 这是一个通用的服务器端错误，请联系Python服务维护人员。 |
