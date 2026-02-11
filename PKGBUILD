# Maintainer: Your Name <your@email.com>
pkgname=reeder
pkgver=0.1.0
pkgrel=1
pkgdesc="Personal TTS RSS service - convert articles to audio podcasts"
arch=('any')
url="https://github.com/youruser/reeder"
license=('MIT')
depends=('python>=3.11' 'uv' 'caddy' 'ffmpeg' 'sox')
backup=('etc/reeder/config.toml')
install=reeder.install
source=()

package() {
    cd "$startdir"

    # Install binaries
    install -Dm755 bin/process-job "$pkgdir/usr/lib/reeder/bin/process-job"
    install -Dm755 bin/update-feed "$pkgdir/usr/lib/reeder/bin/update-feed"
    install -Dm755 bin/reeder-web "$pkgdir/usr/lib/reeder/bin/reeder-web"
    install -Dm755 bin/submit-url "$pkgdir/usr/lib/reeder/bin/submit-url"
    install -Dm755 bin/submit-text "$pkgdir/usr/lib/reeder/bin/submit-text"
    install -Dm755 bin/reeder-status "$pkgdir/usr/lib/reeder/bin/reeder-status"

    # Install Python project files
    install -Dm644 pyproject.toml "$pkgdir/usr/lib/reeder/pyproject.toml"
    install -Dm644 uv.lock "$pkgdir/usr/lib/reeder/uv.lock"

    # Install HTML templates
    install -Dm644 templates/index.html "$pkgdir/usr/lib/reeder/templates/index.html"
    install -Dm644 templates/bookmarklet.html "$pkgdir/usr/lib/reeder/templates/bookmarklet.html"

    # Install config
    install -Dm644 config.toml "$pkgdir/etc/reeder/config.toml"

    # Install Caddyfile
    install -Dm644 Caddyfile "$pkgdir/etc/reeder/Caddyfile"

    # Install systemd units
    install -Dm644 systemd-pkg/reeder.service "$pkgdir/usr/lib/systemd/system/reeder.service"
    install -Dm644 systemd-pkg/reeder-web.service "$pkgdir/usr/lib/systemd/system/reeder-web.service"
    install -Dm644 systemd-pkg/reeder.path "$pkgdir/usr/lib/systemd/system/reeder.path"

    # Install docs
    install -Dm644 README.md "$pkgdir/usr/share/doc/reeder/README.md"
    install -Dm644 LICENSE "$pkgdir/usr/share/licenses/reeder/LICENSE"

    # Install the install script
    install -Dm644 reeder.install "$pkgdir/usr/share/reeder/reeder.install"
}
