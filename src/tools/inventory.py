products = {
    "A100": {"name": "Industrial Sensor A100", "price": 100, "stock": 20},
    "B200": {"name": "Industrial Sensor B200", "price": 180, "stock": 8},
}


def get_inventory(product_id: str) -> dict | None:
    """直接调用（不经过模型）时的行为：商品存在就返回完整记录，不存在就返回
    None，不抛异常。这是刻意的——Commit 1.3 要验证的是"模型拿到'没找到'这个
    信号之后会不会诚实说明"，如果这一层直接抛异常，整条链路会在到达模型之前
    就被打断，反而测不到真正想测的行为。"""
    return products.get(product_id)


def add_inventory(product_id: str, qty: int) -> dict | None:
    """入库：库存数量增加 qty。这一版故意不做任何校验（qty 是不是正数、
    product_id 存不存在都不检查，找不到直接返回 None）——参数校验是 Phase 3
    的范围，这里先只管"写操作本身怎么接进 Tool Calling 的流程"这一件事。"""
    product = products.get(product_id)
    if product is None:
        return None
    product["stock"] += qty
    return product


def remove_inventory(product_id: str, qty: int) -> dict | None:
    """出库：库存数量减少 qty。和 add_inventory 一样，这一版不做任何校验——
    库存可以被减成负数，qty 可以传负数，这些"业务上不合理但代码不会报错"
    的情况，都留给 Phase 3 的 Tool Validation 去处理，不在这里提前解决。"""
    product = products.get(product_id)
    if product is None:
        return None
    product["stock"] -= qty
    return product


# 符合 OpenAI Function Calling 协议形状的 Tool Schema（DeepSeek 用的就是这一套，
# 见 Commit 0.1）。description 是直接喂给模型的输入，不是写给人看的代码注释——
# 模型会不会在正确的时机决定调用这个工具，很大程度上取决于这段文字写得好不好。
GET_INVENTORY_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_inventory",
        "description": (
            "Look up the current stock level and unit price for a product, "
            "given its product ID. Use this whenever the user asks about "
            "stock, availability, or price for a specific product."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "product_id": {
                    "type": "string",
                    "description": "The product ID, e.g. 'A100' or 'B200'.",
                },
            },
            "required": ["product_id"],
        },
    },
}

ADD_INVENTORY_SCHEMA = {
    "type": "function",
    "function": {
        "name": "add_inventory",
        "description": (
            "Add stock for a product (e.g. after a new shipment arrives). "
            "Use this when the user asks to add, restock, or increase "
            "inventory for a specific product."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "product_id": {
                    "type": "string",
                    "description": "The product ID, e.g. 'A100' or 'B200'.",
                },
                "qty": {
                    "type": "integer",
                    "description": "How many units to add to stock.",
                },
            },
            "required": ["product_id", "qty"],
        },
    },
}

REMOVE_INVENTORY_SCHEMA = {
    "type": "function",
    "function": {
        "name": "remove_inventory",
        "description": (
            "Remove stock for a product (e.g. after a sale or shipment out). "
            "Use this when the user asks to remove, deduct, or decrease "
            "inventory for a specific product."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "product_id": {
                    "type": "string",
                    "description": "The product ID, e.g. 'A100' or 'B200'.",
                },
                "qty": {
                    "type": "integer",
                    "description": "How many units to remove from stock.",
                },
            },
            "required": ["product_id", "qty"],
        },
    },
}
