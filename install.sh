#!/usr/bin/env bash
# ============================================================================
# Selphi v1.5.3 — Installer
# ============================================================================
# Installs Selphi and all its dependencies (htslib, bcftools, pbwt, Python
# packages) into a self-contained prefix directory.
#
# Usage:
#   ./install.sh                    # install to ./selphi_env
#   ./install.sh /opt/selphi        # install to custom prefix
#   ./install.sh --help
#
# After installation, run Selphi with:
#   /path/to/selphi_env/bin/selphi --help
#
# Supports: Ubuntu/Debian, CentOS/RHEL/Fedora, macOS (with Homebrew)
# ============================================================================

set -euo pipefail

# ----- defaults -------------------------------------------------------------
PREFIX=""
JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
SKIP_PYTHON=0
SKIP_XSQUEEZEIT=0
PYTHON_CMD=""

print_help() {
    cat <<'EOF'
Selphi v1.5.3 Installer

Usage:
  ./install.sh [OPTIONS] [PREFIX]

Arguments:
  PREFIX              Installation directory (default: ./selphi_env)

Options:
  -j, --jobs N        Parallel build jobs (default: auto-detect)
  --python PYTHON     Path to Python 3.10+ executable (default: auto-detect)
  --skip-xsqueezeit   Skip xSqueezeIt build (only needed for .xsi ref panels)
  --skip-python       Skip Python environment setup (use system Python)
  -h, --help          Show this help

Examples:
  ./install.sh                          # Quick install to ./selphi_env
  ./install.sh /opt/selphi              # Install to /opt/selphi
  ./install.sh --python python3.11 .    # Use specific Python, install in cwd
  ./install.sh --skip-xsqueezeit        # Skip optional xSqueezeIt
EOF
    exit 0
}

# ----- parse arguments ------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)       print_help ;;
        -j|--jobs)       JOBS="$2"; shift 2 ;;
        --python)        PYTHON_CMD="$2"; shift 2 ;;
        --skip-python)   SKIP_PYTHON=1; shift ;;
        --skip-xsqueezeit) SKIP_XSQUEEZEIT=1; shift ;;
        -*)              echo "Unknown option: $1"; print_help ;;
        *)               PREFIX="$1"; shift ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="${PREFIX:-${SCRIPT_DIR}/selphi_env}"
PREFIX="$(mkdir -p "$PREFIX" && cd "$PREFIX" && pwd)"

# ----- logging --------------------------------------------------------------
log()  { echo -e "\033[1;32m[selphi-install]\033[0m $*"; }
warn() { echo -e "\033[1;33m[selphi-install WARNING]\033[0m $*"; }
err()  { echo -e "\033[1;31m[selphi-install ERROR]\033[0m $*" >&2; exit 1; }

log "Installing Selphi v1.5.3 to: $PREFIX"
log "Source directory: $SCRIPT_DIR"
log "Build jobs: $JOBS"

# ----- detect OS ------------------------------------------------------------
detect_os() {
    if [[ -f /etc/os-release ]]; then
        . /etc/os-release
        case "$ID" in
            ubuntu|debian|pop|linuxmint)  echo "debian" ;;
            centos|rhel|rocky|alma|fedora|amzn) echo "rhel" ;;
            arch|manjaro)                 echo "arch" ;;
            *)                            echo "unknown-linux" ;;
        esac
    elif [[ "$(uname -s)" == "Darwin" ]]; then
        echo "macos"
    else
        echo "unknown"
    fi
}

OS=$(detect_os)
log "Detected OS family: $OS"

# ----- check system dependencies -------------------------------------------
check_system_deps() {
    local missing=()

    # Check for C compiler
    if ! command -v gcc &>/dev/null && ! command -v cc &>/dev/null; then
        missing+=("C compiler (gcc)")
    fi

    # Check for make
    if ! command -v make &>/dev/null; then
        missing+=("make")
    fi

    # Check for autoconf
    if ! command -v autoconf &>/dev/null; then
        missing+=("autoconf")
    fi

    # Check for autoreconf (part of autoconf on most systems)
    if ! command -v autoreconf &>/dev/null; then
        missing+=("autoconf (autoreconf)")
    fi

    # Check for git
    if ! command -v git &>/dev/null; then
        missing+=("git")
    fi

    # Check for required -dev headers
    local -A headers=(
        ["zlib"]="zlib.h"
        ["libbz2"]="bzlib.h"
        ["liblzma"]="lzma.h"
        ["libzip"]="zip.h"
        ["libcurl"]="curl/curl.h"
    )
    for lib in "${!headers[@]}"; do
        local hdr="${headers[$lib]}"
        local found=0
        local search_dirs=(/usr/include /usr/local/include /usr/include/x86_64-linux-gnu /usr/include/aarch64-linux-gnu)
        # macOS: Homebrew and Xcode SDK paths
        if [[ "$OS" == "macos" ]]; then
            local brew_prefix
            brew_prefix="$(brew --prefix 2>/dev/null || echo /opt/homebrew)"
            local sdk_path
            sdk_path="$(xcrun --show-sdk-path 2>/dev/null || echo "")"
            search_dirs+=("$brew_prefix/include" "$brew_prefix/opt/curl/include")
            [[ -n "$sdk_path" ]] && search_dirs+=("$sdk_path/usr/include")
        fi
        for d in "${search_dirs[@]}"; do
            if [[ -f "$d/$hdr" ]]; then found=1; break; fi
        done
        if [[ $found -eq 0 ]]; then
            missing+=("$lib development headers ($hdr)")
        fi
    done

    if [[ ${#missing[@]} -gt 0 ]]; then
        warn "Missing system dependencies: ${missing[*]}"
        echo ""
        echo "Install them with:"
        case "$OS" in
            debian)
                echo "  sudo apt-get update && sudo apt-get install -y \\"
                echo "    gcc g++ make autoconf automake git curl pkg-config \\"
                echo "    zlib1g-dev libbz2-dev liblzma-dev libzip-dev libcurl4-openssl-dev"
                ;;
            rhel)
                echo "  sudo yum groupinstall -y 'Development Tools'"
                echo "  sudo yum install -y autoconf automake git curl zlib-devel \\"
                echo "    bzip2-devel xz-devel libzip-devel libcurl-devel"
                ;;
            arch)
                echo "  sudo pacman -S base-devel autoconf automake git curl \\"
                echo "    zlib bzip2 xz libzip"
                ;;
            macos)
                echo "  xcode-select --install"
                echo "  brew install autoconf automake git curl pkg-config xz libzip"
                ;;
            *)
                echo "  Please install: gcc, g++, make, autoconf, git, and development"
                echo "  headers for zlib, bzip2, lzma, libzip, and libcurl."
                ;;
        esac
        echo ""
        read -p "Continue anyway? Some builds may fail. [y/N] " -r
        [[ "$REPLY" =~ ^[Yy]$ ]] || exit 1
    fi
}

check_system_deps

# ----- macOS Homebrew paths -------------------------------------------------
BREW_PREFIX=""
EXTRA_CPPFLAGS=""
EXTRA_LDFLAGS=""
if [[ "$OS" == "macos" ]]; then
    BREW_PREFIX="$(brew --prefix 2>/dev/null || echo /opt/homebrew)"
    EXTRA_CPPFLAGS="-I$BREW_PREFIX/include"
    EXTRA_LDFLAGS="-L$BREW_PREFIX/lib"
    export PKG_CONFIG_PATH="$BREW_PREFIX/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
    log "Homebrew prefix: $BREW_PREFIX"
fi

# ----- directory structure --------------------------------------------------
mkdir -p "$PREFIX"/{bin,lib,include,src,share/selphi}

# ----- build htslib ---------------------------------------------------------
build_htslib() {
    log "Building htslib..."
    local src="$PREFIX/src/htslib"

    if [[ -f "$PREFIX/lib/libhts.a" ]]; then
        log "htslib already built, skipping (remove $PREFIX/lib/libhts.a to rebuild)"
        return
    fi

    if [[ -d "$src" ]]; then
        rm -rf "$src"
    fi

    cd "$PREFIX/src"
    git clone --depth 1 https://github.com/samtools/htslib.git
    cd htslib
    git submodule update --init --recursive
    autoreconf -i
    ./configure --prefix="$PREFIX" --disable-libcurl --enable-plugins=no --without-libdeflate \
        CPPFLAGS="$EXTRA_CPPFLAGS" LDFLAGS="$EXTRA_LDFLAGS"
    make -j"$JOBS"
    make install
    log "htslib installed to $PREFIX"
}

# ----- build bcftools -------------------------------------------------------
build_bcftools() {
    log "Building bcftools..."
    local src="$PREFIX/src/bcftools"

    if [[ -x "$PREFIX/bin/bcftools" ]]; then
        log "bcftools already built, skipping (remove $PREFIX/bin/bcftools to rebuild)"
        return
    fi

    if [[ -d "$src" ]]; then
        rm -rf "$src"
    fi

    cd "$PREFIX/src"
    git clone --depth 1 https://github.com/samtools/bcftools.git
    cd bcftools
    autoheader && autoconf
    # Link against htslib source tree (static) to avoid shared lib version issues
    ./configure --prefix="$PREFIX" \
        CPPFLAGS="-I$PREFIX/include $EXTRA_CPPFLAGS" \
        LDFLAGS="-L$PREFIX/lib $EXTRA_LDFLAGS" \
        --with-htslib="$PREFIX/src/htslib"
    make -j"$JOBS"
    make install
    log "bcftools installed to $PREFIX/bin/bcftools"
}

# ----- build xSqueezeIt (optional) ------------------------------------------
build_xsqueezeit() {
    if [[ $SKIP_XSQUEEZEIT -eq 1 ]]; then
        log "Skipping xSqueezeIt (--skip-xsqueezeit)"
        return
    fi

    log "Building xSqueezeIt (optional, for .xsi reference panels)..."
    local src="$PREFIX/src/xSqueezeIt"

    if [[ -x "$PREFIX/bin/xsqueezeit" ]]; then
        log "xSqueezeIt already built, skipping"
        return
    fi

    if [[ -d "$src" ]]; then
        rm -rf "$src"
    fi

    cd "$PREFIX/src"
    git clone --depth 1 https://github.com/rwk-unil/xSqueezeIt.git
    cd xSqueezeIt
    git submodule update --init --recursive htslib
    cd htslib
    autoheader 2>/dev/null || true
    autoconf 2>/dev/null || true
    automake --add-missing 2>/dev/null || true
    ./configure --prefix="$PREFIX"
    make -j"$JOBS"
    make install
    ldconfig 2>/dev/null || true
    cd ..
    git clone --depth 1 https://github.com/facebook/zstd.git
    cd zstd
    make -j"$JOBS"
    cd ..
    make -j"$JOBS"
    chmod +x xsqueezeit
    cp xsqueezeit "$PREFIX/bin/"
    log "xSqueezeIt installed to $PREFIX/bin/xsqueezeit"
}

# ----- build pbwt -----------------------------------------------------------
build_pbwt() {
    log "Building pbwt library..."

    if [[ -x "$PREFIX/bin/pbwt" ]]; then
        log "pbwt already built, skipping (remove $PREFIX/bin/pbwt to rebuild)"
        return
    fi

    local pbwt_src="$SCRIPT_DIR/pbwt"
    if [[ ! -d "$pbwt_src" ]]; then
        err "pbwt source directory not found at $pbwt_src"
    fi

    # Build in a temporary copy to avoid polluting the source tree
    local build_dir="$PREFIX/src/pbwt_build"
    rm -rf "$build_dir"
    cp -r "$pbwt_src" "$build_dir"
    cd "$build_dir"

    # htslib source needed for headers and libhts.a
    local htsdir="$PREFIX/src/htslib"
    if [[ ! -f "$htsdir/libhts.a" && -f "$PREFIX/lib/libhts.a" ]]; then
        # If source was cleaned, point to installed location
        htsdir="$PREFIX"
    fi

    HTSDIR="$htsdir" make -j"$JOBS"
    cp pbwt "$PREFIX/bin/pbwt"
    chmod +x "$PREFIX/bin/pbwt"
    log "pbwt installed to $PREFIX/bin/pbwt"
}

# ----- detect Python --------------------------------------------------------
find_python() {
    if [[ -n "$PYTHON_CMD" ]]; then
        if command -v "$PYTHON_CMD" &>/dev/null; then
            echo "$PYTHON_CMD"
            return
        else
            err "Specified Python '$PYTHON_CMD' not found"
        fi
    fi

    # Try common Python commands in preference order
    local candidates=("python3.12" "python3.11" "python3.10" "python3" "python")
    for cmd in "${candidates[@]}"; do
        if command -v "$cmd" &>/dev/null; then
            local ver
            ver=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
            local major minor
            major=$(echo "$ver" | cut -d. -f1)
            minor=$(echo "$ver" | cut -d. -f2)
            if [[ "$major" -ge 3 && "$minor" -ge 10 ]]; then
                echo "$cmd"
                return
            fi
        fi
    done

    return 1
}

# ----- setup Python environment ---------------------------------------------
setup_python() {
    if [[ $SKIP_PYTHON -eq 1 ]]; then
        log "Skipping Python setup (--skip-python)"
        return
    fi

    local python
    python=$(find_python) || {
        err "Python 3.10+ not found. Install it or specify with --python.

On Ubuntu/Debian:
  sudo apt-get install -y python3 python3-pip python3-venv

On CentOS/RHEL:
  sudo yum install -y python3 python3-pip

On macOS:
  brew install python@3.11

Or use conda/pyenv to install Python 3.10+."
    }

    local pyver
    pyver=$("$python" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
    log "Using Python: $python ($pyver)"

    # Create virtual environment
    log "Creating Python virtual environment..."
    "$python" -m venv "$PREFIX/venv" 2>/dev/null || {
        # venv module might not be installed
        warn "python3-venv not available, trying virtualenv..."
        if command -v virtualenv &>/dev/null; then
            virtualenv -p "$python" "$PREFIX/venv"
        else
            "$python" -m pip install --user virtualenv 2>/dev/null || true
            "$python" -m virtualenv "$PREFIX/venv" 2>/dev/null || {
                err "Cannot create virtual environment. Install python3-venv:
  Ubuntu/Debian: sudo apt-get install -y python3-venv
  CentOS/RHEL:   sudo yum install -y python3-virtualenv"
            }
        fi
    }

    # Activate and install dependencies
    source "$PREFIX/venv/bin/activate"

    log "Upgrading pip..."
    pip install --upgrade pip setuptools wheel 2>&1 | tail -1

    log "Installing Python dependencies..."
    pip install \
        cachetools \
        "cyvcf2==0.31.1" \
        joblib \
        numba \
        numpy \
        pandas \
        scipy \
        tqdm \
        zstd \
        natsort \
        2>&1 | tail -5

    deactivate
    log "Python environment ready at $PREFIX/venv"
}

# ----- install Selphi files -------------------------------------------------
install_selphi() {
    log "Installing Selphi files..."

    # Copy Python source
    cp "$SCRIPT_DIR/selphi.py" "$PREFIX/share/selphi/"
    cp "$SCRIPT_DIR/pyproject.toml" "$PREFIX/share/selphi/"
    cp -r "$SCRIPT_DIR/modules" "$PREFIX/share/selphi/"

    # Create wrapper script
    cat > "$PREFIX/bin/selphi" <<WRAPPER
#!/usr/bin/env bash
# Selphi v1.5.3 wrapper
# Auto-generated by install.sh

SELF="\$0"
# Resolve symlinks (portable, works on macOS without coreutils)
while [[ -L "\$SELF" ]]; do
    DIR="\$(cd "\$(dirname "\$SELF")" && pwd)"
    SELF="\$(readlink "\$SELF")"
    [[ "\$SELF" != /* ]] && SELF="\$DIR/\$SELF"
done
SELPHI_HOME="\$(cd "\$(dirname "\$SELF")/.." && pwd)"

# Activate virtual environment if present
if [[ -f "\$SELPHI_HOME/venv/bin/activate" ]]; then
    source "\$SELPHI_HOME/venv/bin/activate"
fi

# Add local bin to PATH for bcftools, pbwt, etc.
export PATH="\$SELPHI_HOME/bin:\$PATH"
if [[ "\$(uname -s)" == "Darwin" ]]; then
    export DYLD_LIBRARY_PATH="\$SELPHI_HOME/lib:\${DYLD_LIBRARY_PATH:-}"
else
    export LD_LIBRARY_PATH="\$SELPHI_HOME/lib:\${LD_LIBRARY_PATH:-}"
fi

exec python3 "\$SELPHI_HOME/share/selphi/selphi.py" \\
    --pbwt_path "\$SELPHI_HOME/bin/pbwt" \\
    "\$@"
WRAPPER
    chmod +x "$PREFIX/bin/selphi"
    log "Selphi wrapper installed to $PREFIX/bin/selphi"
}

# ----- verify installation --------------------------------------------------
verify() {
    log "Verifying installation..."
    local ok=1

    # Check binaries
    for bin in bcftools pbwt; do
        if [[ -x "$PREFIX/bin/$bin" ]]; then
            log "  $bin: OK"
        else
            warn "  $bin: MISSING"
            ok=0
        fi
    done

    if [[ $SKIP_XSQUEEZEIT -eq 0 ]]; then
        if [[ -x "$PREFIX/bin/xsqueezeit" ]]; then
            log "  xsqueezeit: OK"
        else
            warn "  xsqueezeit: MISSING (optional — only needed for .xsi reference panels)"
        fi
    fi

    # Check Python environment
    if [[ $SKIP_PYTHON -eq 0 ]]; then
        if [[ -f "$PREFIX/venv/bin/activate" ]]; then
            source "$PREFIX/venv/bin/activate"
            if python3 -c "import cyvcf2, numba, numpy, pandas, scipy, joblib, tqdm" 2>/dev/null; then
                log "  Python packages: OK"
            else
                warn "  Python packages: some imports failed"
                ok=0
            fi
            deactivate
        else
            warn "  Python venv: MISSING"
            ok=0
        fi
    fi

    # Check Selphi wrapper
    if [[ -x "$PREFIX/bin/selphi" ]]; then
        log "  selphi wrapper: OK"
    else
        warn "  selphi wrapper: MISSING"
        ok=0
    fi

    # Check Selphi can show help
    if [[ $SKIP_PYTHON -eq 0 && -x "$PREFIX/bin/selphi" ]]; then
        if "$PREFIX/bin/selphi" -h &>/dev/null; then
            log "  selphi --help: OK"
        else
            warn "  selphi --help: failed"
            ok=0
        fi
    fi

    return $((1 - ok))
}

# ----- main -----------------------------------------------------------------
main() {
    local start_time=$SECONDS

    build_htslib
    build_bcftools
    build_xsqueezeit
    build_pbwt
    setup_python
    install_selphi

    echo ""
    echo "============================================================"
    if verify; then
        log "Installation complete! ($((SECONDS - start_time))s)"
    else
        warn "Installation completed with warnings ($((SECONDS - start_time))s)"
    fi
    echo "============================================================"
    echo ""
    echo "To use Selphi, either:"
    echo ""
    echo "  1. Add to PATH:"
    echo "     export PATH=\"$PREFIX/bin:\$PATH\""
    echo ""
    echo "  2. Or run directly:"
    echo "     $PREFIX/bin/selphi --help"
    echo ""
    echo "Quick start:"
    echo "  # Prepare a reference panel from VCF/BCF:"
    echo "  selphi --prepare_reference --ref_source_vcf ref.bcf --refpanel /path/to/output_prefix"
    echo ""
    echo "  # Run imputation:"
    echo "  selphi --refpanel /path/to/ref_prefix --target target.vcf.gz \\"
    echo "         --map genetic_map.chr1.GRCh38.map --outvcf output --cores 16"
    echo ""
}

main 2>&1 | tee "$PREFIX/install.log"
