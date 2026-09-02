import type { HeadConfig, TransformContext } from 'vitepress'

export const SITE_ORIGIN = 'https://xingwudao.github.io'
export const SITE_BASE = '/open-xquant/'

const SITE_TITLE = 'open-xquant'
const SITE_DESCRIPTION = '面向 AI Agent 和量化研究者的中文友好 AI 量化研究框架'
const REPOSITORY_URL = 'https://github.com/xingwudao/open-xquant'
const LICENSE_URL = 'https://opensource.org/licenses/MIT'
const OG_IMAGE = `${SITE_ORIGIN}${SITE_BASE}images/open-xquant-subagent-collaboration.png`
const OG_IMAGE_ALT = 'open-xquant AI 量化研究框架中 Agent、Skills 与确定性内核的协作架构图'

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
  const jsonLd = jsonLdFor(context.pageData.relativePath, title, description, canonical)

  return [
    ['link', { rel: 'canonical', href: canonical }],
    ['meta', { property: 'og:title', content: title }],
    ['meta', { property: 'og:description', content: description }],
    ['meta', { property: 'og:type', content: 'website' }],
    ['meta', { property: 'og:url', content: canonical }],
    ['meta', { property: 'og:site_name', content: SITE_TITLE }],
    ['meta', { property: 'og:locale', content: 'zh_CN' }],
    ['meta', { property: 'og:image', content: OG_IMAGE }],
    ['meta', { property: 'og:image:alt', content: OG_IMAGE_ALT }],
    ['meta', { name: 'twitter:card', content: 'summary_large_image' }],
    ['meta', { name: 'twitter:title', content: title }],
    ['meta', { name: 'twitter:description', content: description }],
    ['meta', { name: 'twitter:image', content: OG_IMAGE }],
    ['script', { type: 'application/ld+json' }, JSON.stringify(jsonLd)],
  ]
}

function jsonLdFor(
  relativePath: string,
  title: string,
  description: string,
  canonical: string,
): Record<string, unknown>[] {
  const breadcrumb = {
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

  if (relativePath === 'index.md') {
    return [
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
        programmingLanguage: 'Python',
      },
      {
        '@context': 'https://schema.org',
        '@type': 'SoftwareApplication',
        name: SITE_TITLE,
        applicationCategory: 'DeveloperApplication',
        operatingSystem: 'macOS, Linux, Windows',
        description: SITE_DESCRIPTION,
        url: canonical,
        softwareHelp: `${SITE_ORIGIN}${SITE_BASE}guide/ai-quant-framework`,
      },
    ]
  }

  if (relativePath === 'faq/index.md') {
    return [
      breadcrumb,
      {
        '@context': 'https://schema.org',
        '@type': 'FAQPage',
        name: title,
        description,
        url: canonical,
        mainEntity: [
          {
            '@type': 'Question',
            name: 'open-xquant 是 AI 交易机器人吗？',
            acceptedAnswer: {
              '@type': 'Answer',
              text: '不是。open-xquant 是面向 AI Agent 和人类研究者的确定性量化研究内核。',
            },
          },
          {
            '@type': 'Question',
            name: 'AI 生成的量化策略为什么还需要审计？',
            acceptedAnswer: {
              '@type': 'Answer',
              text: '审计用于检查未确认假设、实现漂移、数据泄漏、执行语义和产物血缘。',
            },
          },
        ],
      },
    ]
  }

  if (relativePath === 'skills/index.md' || relativePath === 'tools/index.md') {
    return [
      breadcrumb,
      {
        '@context': 'https://schema.org',
        '@type': 'ItemList',
        name: title,
        description,
        url: canonical,
      },
    ]
  }

  return [
    breadcrumb,
    {
      '@context': 'https://schema.org',
      '@type': 'TechArticle',
      headline: title,
      description,
      url: canonical,
      about: ['AI 量化', '量化回测', '因子研究', 'Agentic Quant Research'],
    },
  ]
}
