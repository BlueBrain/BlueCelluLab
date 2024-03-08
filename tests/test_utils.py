from bluecellulab.utils import CaptureOutput, run_once


# Decorated function for testing
@run_once
def increment_counter(counter):
    counter[0] += 1
    return "Executed"


def test_run_once_execution():
    """Test that the decorated function runs only once."""
    counter = [0]  # Using a list for mutability

    assert increment_counter(counter) == "Executed"
    increment_counter(counter)
    assert counter[0] == 1

    # Called 3 times but increased once
    increment_counter(counter)
    increment_counter(counter)
    increment_counter(counter)

    assert counter[0] == 1

    assert increment_counter(counter) is None


def test_capture_output():
    with CaptureOutput() as output:
        print("Hello, World!")
        print("This is a test.")

    assert len(output) == 2
    assert "Hello, World!" in output
    assert "This is a test." in output


def test_no_output():
    with CaptureOutput() as output:
        pass  # No output

    assert len(output) == 0
