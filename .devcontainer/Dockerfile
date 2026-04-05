# Devcontainer for arrax: Python 3.14 (via uv), RISC-V cross toolchain,
# and the riscv-npu sibling repo — all inside a locked-down sandbox with
# Claude Code pre-installed.
FROM mcr.microsoft.com/devcontainers/base:bookworm

ARG TZ
ENV TZ="$TZ"

ARG USERNAME=vscode

# Install packages needed by arrax's pipeline and by the sandbox firewall.
# The base image already ships git, curl, wget, sudo, zsh + oh-my-zsh, ca-certificates.
RUN apt-get update && apt-get install -y --no-install-recommends \
  less \
  fzf \
  man-db \
  unzip \
  gnupg2 \
  gh \
  iptables \
  ipset \
  iproute2 \
  dnsutils \
  aggregate \
  jq \
  nano \
  vim \
  build-essential \
  make \
  gcc-riscv64-unknown-elf \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

# Persist shell history across container rebuilds via a named volume mount.
RUN mkdir /commandhistory \
  && touch /commandhistory/.bash_history \
  && chown -R $USERNAME /commandhistory \
  && echo "export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" \
     >> /home/$USERNAME/.zshrc \
  && echo "export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" \
     >> /home/$USERNAME/.bashrc

ENV DEVCONTAINER=true

# Workspace, Claude config, and VS Code / Cursor server cache dirs. /vscode is
# a named-volume mountpoint used by the Dev Containers extension to cache the
# server binaries across rebuilds — it must be writable by $USERNAME, otherwise
# Cursor/VS Code hangs during server install.
RUN mkdir -p /workspace /home/$USERNAME/.claude /vscode && \
  chown -R $USERNAME:$USERNAME /workspace /home/$USERNAME/.claude /vscode

WORKDIR /workspace

# git-delta for prettier diffs.
ARG GIT_DELTA_VERSION=0.18.2
RUN ARCH=$(dpkg --print-architecture) && \
  wget "https://github.com/dandavison/delta/releases/download/${GIT_DELTA_VERSION}/git-delta_${GIT_DELTA_VERSION}_${ARCH}.deb" && \
  dpkg -i "git-delta_${GIT_DELTA_VERSION}_${ARCH}.deb" && \
  rm "git-delta_${GIT_DELTA_VERSION}_${ARCH}.deb"

# Clone the riscv-npu sibling repo. pyproject.toml declares it as an editable
# path dependency at `../riscv-npu`, which resolves to /riscv-npu when the
# arrax workspace is mounted at /workspace. Also assemble start.o, which
# arrax's codegen links against at runtime but which riscv-npu distributes
# only as start.s source.
ARG RISCV_NPU_REPO=https://github.com/paladin8/riscv-npu.git
ARG RISCV_NPU_REF=main
RUN git clone --depth 1 --branch "${RISCV_NPU_REF}" "${RISCV_NPU_REPO}" /riscv-npu && \
  make -C /riscv-npu/firmware/common start.o && \
  chown -R $USERNAME:$USERNAME /riscv-npu

USER $USERNAME

ENV SHELL=/bin/zsh
ENV EDITOR=nano
ENV VISUAL=nano

# Install Claude Code (native build lands in ~/.local/bin).
ENV PATH=/home/vscode/.local/bin:$PATH
RUN curl -fsSL https://claude.ai/install.sh | bash

# Install uv and pre-fetch the Python interpreter pinned by pyproject.toml
# (requires-python = ">=3.14"). Both live under the user's home.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
  uv python install 3.14

# Firewall script (executed by postStartCommand in devcontainer.json).
USER root
COPY init-firewall.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/init-firewall.sh && \
  echo "$USERNAME ALL=(root) NOPASSWD: /usr/local/bin/init-firewall.sh" > /etc/sudoers.d/$USERNAME-firewall && \
  chmod 0440 /etc/sudoers.d/$USERNAME-firewall
USER $USERNAME
