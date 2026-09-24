#!/usr/bin/env python3
"""Check or install DuckDB's official Excel extension with an explicit consent gate."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import duckdb

__all__ = []

_EXTENSION_NAME = "excel"


@dataclass(frozen=True)
class _ExtensionState:
    installed: bool
    loaded: bool
    version: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check Excel support without side effects, or install DuckDB's signed core Excel extension "
            "after the user explicitly approves the local download and installation."
        )
    )
    subparsers = parser.add_subparsers(dest="action", required=True)
    subparsers.add_parser(
        "check",
        help="Check whether Excel support is installed. Makes no network request and changes nothing.",
    )
    install_parser = subparsers.add_parser(
        "install",
        help="Install and load the signed Excel extension from DuckDB's official core repository.",
    )
    install_parser.add_argument(
        "--accept-local-install",
        action="store_true",
        help="Confirm that the user approved the explained network download and persistent local installation.",
    )
    return parser.parse_args()


def _configure_extension_security(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("SET autoinstall_known_extensions = false")
    conn.execute("SET allow_community_extensions = false")
    conn.execute("SET allow_unsigned_extensions = false")


def _extension_state(conn: duckdb.DuckDBPyConnection) -> _ExtensionState:
    row = conn.execute(
        """
        SELECT installed, loaded, extension_version
        FROM duckdb_extensions()
        WHERE extension_name = ?
        """,
        [_EXTENSION_NAME],
    ).fetchone()
    if row is None:
        raise RuntimeError("The installed DuckDB version does not list the Excel extension.")
    return _ExtensionState(
        installed=bool(row[0]),
        loaded=bool(row[1]),
        version=str(row[2] or ""),
    )


def _print_check(state: _ExtensionState) -> None:
    if state.installed:
        print("Excel support: installed")
        print("Network request: not performed")
        print("Local changes: none")
        if state.version:
            print(f"Extension version: {state.version}")
        return
    print("Excel support: not installed")
    print("Network request: not performed")
    print("Local changes: none")
    print("Next step: explain the installation impact and options, then ask the user for a choice.")


def _print_consent_required() -> None:
    print("Installation was not started because explicit user consent was not recorded.", file=sys.stderr)
    print("Explain these points before asking:", file=sys.stderr)
    print("- Downloads the signed Excel support component from its official repository.", file=sys.stderr)
    print("- Stores it in the configured local extension area so later sessions can reuse it.", file=sys.stderr)
    print("- The component runs with the same local permissions as this process.", file=sys.stderr)
    print("- Does not open, modify, or upload any workbook during installation.", file=sys.stderr)
    print("- Uses network access and some local disk space; organizational policy may block it.", file=sys.stderr)
    print("Alternatives: export the required sheet to CSV/Parquet, or stop.", file=sys.stderr)
    print("After the user approves installation, rerun with --accept-local-install.", file=sys.stderr)


def _install() -> int:
    try:
        with duckdb.connect() as conn:
            _configure_extension_security(conn)
            before = _extension_state(conn)
            downloaded = not before.installed
            if downloaded:
                conn.execute("INSTALL excel FROM core")
            conn.execute("LOAD excel")
            after = _extension_state(conn)
    except (duckdb.Error, RuntimeError) as exc:
        print("Excel support could not be prepared.", file=sys.stderr)
        print(f"Technical detail: {exc}", file=sys.stderr)
        print("No workbook was opened or modified.", file=sys.stderr)
        print("Offer CSV/Parquet export or ask the environment administrator for help.", file=sys.stderr)
        return 1

    if not after.installed or not after.loaded:
        print("Excel support did not reach the required installed-and-loaded state.", file=sys.stderr)
        return 1

    print("Excel support: ready")
    print(f"Official download performed: {'yes' if downloaded else 'no; it was already installed'}")
    print("Signature policy: official core component only")
    print("Workbook access: none")
    print("Workbook changes: none")
    if after.version:
        print(f"Extension version: {after.version}")
    return 0


def main() -> int:
    args = _parse_args()
    if args.action == "check":
        try:
            with duckdb.connect() as conn:
                _configure_extension_security(conn)
                state = _extension_state(conn)
        except (duckdb.Error, RuntimeError) as exc:
            print(f"Excel support could not be checked: {exc}", file=sys.stderr)
            return 1
        _print_check(state)
        return 0

    if not args.accept_local_install:
        _print_consent_required()
        return 2
    return _install()


if __name__ == "__main__":
    raise SystemExit(main())
