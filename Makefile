.PHONY: package deploy clean dev

# Build Arch package on remote host
package:
	@if [ -z "$$DEPLOY_HOST" ]; then echo "Error: DEPLOY_HOST not set"; exit 1; fi
	ssh "$$DEPLOY_HOST" 'rm -rf /tmp/reeder-build'
	rsync -av --exclude='.git' --exclude='.venv' \
		--exclude='inbox/' --exclude='processing/' --exclude='done/' \
		--exclude='www/audio/' --exclude='var/' --exclude='*.pkg.tar.zst' \
		./ "$$DEPLOY_HOST:/tmp/reeder-build/"
	ssh "$$DEPLOY_HOST" 'cd /tmp/reeder-build && makepkg -fd'
	scp "$$DEPLOY_HOST:/tmp/reeder-build/reeder-*.pkg.tar.zst" ./

# Deploy to remote host (set DEPLOY_HOST env var)
deploy: package
	@if [ -z "$$DEPLOY_HOST" ]; then echo "Error: DEPLOY_HOST not set"; exit 1; fi
	scp reeder-*.pkg.tar.zst "$$DEPLOY_HOST:/tmp/"
	ssh -t "$$DEPLOY_HOST" 'sudo pacman -U --noconfirm /tmp/reeder-*.pkg.tar.zst && rm /tmp/reeder-*.pkg.tar.zst'

# Clean build artifacts
clean:
	rm -rf pkg/ src/ *.pkg.tar.zst *.tar.gz

# Local development
dev:
	./dev-setup.sh
