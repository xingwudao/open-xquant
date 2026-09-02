import type { HeadConfig, TransformContext } from 'vitepress'

export const SITE_ORIGIN = 'https://xingwudao.github.io'
export const SITE_BASE = '/open-xquant/'

const SITE_TITLE = 'open-xquant'
const SITE_DESCRIPTION = '面向 AI Agent 和量化研究者的中文友好 AI 量化研究框架'
const REPOSITORY_URL = 'https://github.com/xingwudao/open-xquant'
const LICENSE_URL = 'https://opensource.org/licenses/MIT'
const OG_IMAGE = `${SITE_ORIGIN}${SITE_BASE}images/open-xquant-subagent-collaboration.png`

export function canonicalFor(relativePath: string): string {
  const normalized = relativePath.replace(/\\/g, '/').replace(/^\.\//, '')
  const pagePath = normalized.endsWith('index.md')
    ? normalized.slice(0, -'index.md'.length)
    : normalized.replace(/\.md$/, '')
  return new URL(`${SITE_BASE}${pagePath.replace(/^\/+/, '')}`, SITE_ORIGIN).toString()
}

export function seoHead(context: TransformContext): HeadConfig[] {
  const canonical = canonicalFor(context.pageData.relativePath)
  const title = context.pageData.title || SITE_TITLE
  const description = context.pageData.description || SITE_DESCRIPTION
  const jsonLd =
    context.pageData.relativePath === 'index.md'
      ? [
          {
            '@context': 'https://schema.org',
            '@type': 'WebSite',
            name: SITE_TITLE,
            description,
            url: canonical,
          },
          {
            '@context': 'https://schema.org',
            '@type': 'SoftwareSourceCode',
            name: SITE_TITLE,
            description: SITE_DESCRIPTION,
            url: canonical,
            codeRepository: REPOSITORY_URL,
            license: LICENSE_URL,
          },
        ]
      : {
          '@context': 'https://schema.org',
          '@type': 'BreadcrumbList',
          itemListElement: [
            {
              '@type': 'ListItem',
              position: 1,
              name: SITE_TITLE,
              item: `${SITE_ORIGIN}${SITE_BASE}`,
            },
            {
              '@type': 'ListItem',
              position: 2,
              name: title,
              item: canonical,
            },
          ],
        }

  return [
    ['link', { rel: 'canonical', href: canonical }],
    ['meta', { property: 'og:title', content: title }],
    ['meta', { property: 'og:description', content: description }],
    ['meta', { property: 'og:type', content: 'website' }],
    ['meta', { property: 'og:url', content: canonical }],
    ['meta', { property: 'og:locale', content: 'zh_CN' }],
    ['meta', { property: 'og:image', content: OG_IMAGE }],
    ['script', { type: 'application/ld+json' }, JSON.stringify(jsonLd)],
  ]
}
