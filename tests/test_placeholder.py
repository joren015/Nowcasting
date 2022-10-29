import pytest

from nowcasting.placeholder import placeholder_func


def test_placeholder() -> None:
	assert placeholder_func() == "Hello world!"
