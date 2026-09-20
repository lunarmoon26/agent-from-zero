import copy

import pytest

from src.tools import inventory
from src.tools.inventory import (
    ADD_INVENTORY_SCHEMA,
    GET_INVENTORY_SCHEMA,
    REMOVE_INVENTORY_SCHEMA,
    add_inventory,
    get_inventory,
    products,
    remove_inventory,
)


@pytest.fixture(autouse=True)
def _restore_products():
    """add_inventory/remove_inventory 会真的修改模块级的 products 字典——
    没有这个 fixture，测试跑的先后顺序会互相污染彼此的库存数字。"""
    snapshot = copy.deepcopy(products)
    yield
    products.clear()
    products.update(snapshot)


def test_get_inventory_returns_the_full_record_for_a_known_product():
    result = get_inventory("A100")

    assert result == {"name": "Industrial Sensor A100", "price": 100, "stock": 20}


def test_get_inventory_returns_none_for_an_unknown_product():
    result = get_inventory("Z999")

    assert result is None


def test_products_data_matches_the_documented_scenario():
    assert products["A100"]["stock"] == 20
    assert products["B200"]["stock"] == 8


def test_schema_has_the_shape_the_openai_compatible_api_expects():
    assert GET_INVENTORY_SCHEMA["type"] == "function"

    function = GET_INVENTORY_SCHEMA["function"]
    assert function["name"] == "get_inventory"
    assert isinstance(function["description"], str) and len(function["description"]) > 0

    params = function["parameters"]
    assert params["type"] == "object"
    assert "product_id" in params["properties"]
    assert params["properties"]["product_id"]["type"] == "string"
    assert params["required"] == ["product_id"]


class TestAddInventory:
    def test_increases_stock_by_qty(self):
        before = products["A100"]["stock"]

        result = add_inventory("A100", 5)

        assert result["stock"] == before + 5
        assert products["A100"]["stock"] == before + 5

    def test_returns_none_for_unknown_product(self):
        assert add_inventory("Z999", 5) is None

    def test_schema_requires_product_id_and_qty(self):
        params = ADD_INVENTORY_SCHEMA["function"]["parameters"]
        assert set(params["required"]) == {"product_id", "qty"}
        assert params["properties"]["qty"]["type"] == "integer"


class TestRemoveInventory:
    def test_decreases_stock_by_qty(self):
        before = products["A100"]["stock"]

        result = remove_inventory("A100", 5)

        assert result["stock"] == before - 5
        assert products["A100"]["stock"] == before - 5

    def test_returns_none_for_unknown_product(self):
        assert remove_inventory("Z999", 5) is None

    def test_schema_requires_product_id_and_qty(self):
        params = REMOVE_INVENTORY_SCHEMA["function"]["parameters"]
        assert set(params["required"]) == {"product_id", "qty"}
        assert params["properties"]["qty"]["type"] == "integer"

    def test_does_not_validate_quantity_yet(self):
        # 这一版故意不做校验（Phase 3 才做）——负数、超库存都会被"允许"，
        # 这条测试的意义是把这个当前行为钉死，Phase 3 改了之后这条测试
        # 应该会失败，提醒我们同步更新这份记录。
        result = remove_inventory("B200", 999)

        assert result["stock"] == inventory.products["B200"]["stock"]
        assert result["stock"] < 0
