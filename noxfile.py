import nox

nox.options.sessions = ["lint", "tests"]


@nox.session(python=["3.11"])
def tests(session: nox.Session) -> None:
    """
    Run the unit and regular tests.
    """
    session.install("-e.[test]")
    session.run("pytest", *session.posargs)


@nox.session
def lint(session: nox.Session) -> None:
    """
    Run the linter.
    """
    session.install("pre-commit")
    session.run(
        "pre-commit", "run", "--show-diff-on-failure", "--all-files", *session.posargs
    )
